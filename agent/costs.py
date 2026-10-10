"""Session-local estimates from provider usage, not an account billing balance."""
import time
import json
import os
import uuid
from pathlib import Path

PRICING = dict(
    model='gpt-6-astra', currency='USD', service_tier='default',
    input_per_million=10.0, cached_per_million=1.0,
    cache_write_per_million=12.5, output_per_million=50.0,
    long_context_threshold=272_000, long_input_multiplier=2.0, long_output_multiplier=1.5,
    verified_on='2026-09-11',
    source='https://developers.openai.com/api/docs/models/gpt-6-astra',
)
RATES = {'gpt-6-astra': PRICING, 'gpt-6-sol': dict(
    PRICING, model='gpt-6-sol', input_per_million=2.0, cached_per_million=.2,
    cache_write_per_million=2.5, output_per_million=10.0, verified_on='2026-09-23',
    source='https://developers.openai.com/api/docs/models/gpt-6-sol')}


def rates(model):
    return next((r for name, r in RATES.items() if isinstance(model, str) and (model == name or model.startswith(name + '-'))), None)


def count(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def price_response(response):
    """Missing/invalid usage is unknown, never silently priced as zero."""
    response = response if isinstance(response, dict) else {}
    usage = response.get('usage') or {}
    usage = usage if isinstance(usage, dict) else {}
    details = usage.get('input_tokens_details') or {}
    details = details if isinstance(details, dict) else {}
    inp, out = count(usage.get('input_tokens')), count(usage.get('output_tokens'))
    cached, written = count(details.get('cached_tokens')), count(details.get('cache_write_tokens'))
    result = dict(input_tokens=inp, output_tokens=out, cached_tokens=cached,
                  cache_write_tokens=written, usd_low=None, usd_high=None,
                  long_context=False, note=None)
    output_details = usage.get('output_tokens_details') or {}
    result['reasoning_tokens'] = count(output_details.get('reasoning_tokens')) if isinstance(output_details, dict) else None
    result['total_tokens'] = count(usage.get('total_tokens'))
    if result['total_tokens'] is None and inp is not None and out is not None:
        result['total_tokens'] = inp + out
    PRICING = rates(response.get('model', ''))
    if PRICING is None:
        result['note'] = 'No verified rate for the returned model'
        return result
    if response.get('service_tier', 'default') not in ('default', None):
        result['note'] = 'Returned service tier has no configured rate'
        return result
    if inp is None or out is None or cached is None or cached > inp:
        result['note'] = 'Token usage is missing or invalid'
        return result
    if 'cache_write_tokens' in details and (written is None or cached + written > inp):
        result['note'] = 'Cache-write usage is invalid'
        return result
    long_context = inp > PRICING['long_context_threshold']
    mi, mo = (PRICING['long_input_multiplier'], PRICING['long_output_multiplier']) if long_context else (1, 1)
    ordinary_or_write = inp - cached
    base = (ordinary_or_write * PRICING['input_per_million'] * mi
            + cached * PRICING['cached_per_million'] * mi
            + out * PRICING['output_per_million'] * mo) / 1e6
    write_surcharge = PRICING['cache_write_per_million'] - PRICING['input_per_million']
    low_write, high_write = (0, ordinary_or_write) if written is None else (written, written)
    result.update(usd_low=round(base + low_write * write_surcharge * mi / 1e6, 10),
                  usd_high=round(base + high_write * write_surcharge * mi / 1e6, 10),
                  long_context=long_context,
                  note='Cache-write count missing; displaying a range' if written is None else None)
    return result


class CostMeter:
    def __init__(self, model, session_id=None, ledger_path=None):
        self.model = model
        self.requests = []
        self.session_id = session_id
        self.ledger_path = Path(ledger_path) if ledger_path is not None else None

    def persist(self, item):
        if self.ledger_path is None:
            return
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        event = dict(schema_version=1, session_id=self.session_id, recorded_at=time.time(),
                     request=item, pricing=rates(item.get('model')))
        with self.ledger_path.open('a') as f:
            f.write(json.dumps(event, allow_nan=False) + '\n')
            f.flush()
            os.fsync(f.fileno())

    def begin(self, context=None):
        item = dict(number=len(self.requests) + 1, time=time.time(), status='pending',
                    request_id=uuid.uuid4().hex, context=context or {},
                    model=self.model, response_id=None, **price_response(None))
        item['note'] = 'Awaiting provider usage'
        self.persist(item)  # Fail before dispatch if the billing ledger cannot be written.
        self.requests.append(item)
        return item

    def finish(self, item, response=None, status='completed'):
        if item['status'] != 'pending':
            return
        response = response if isinstance(response, dict) else {}
        item.update(price_response(response), status=status,
                    model=response.get('model', self.model), response_id=response.get('id'))
        if not response:
            item['note'] = 'No final usage received; this request may still be billed'
        self.persist(item)

    def export(self):
        return dict(schema_version=1, session_id=self.session_id,
                    **{**self.summary(), 'requests': self.requests[:]})

    def summary(self):
        priced = [r for r in self.requests if r['usd_low'] is not None]
        unknown = sum(r['status'] != 'pending' and r['usd_low'] is None for r in self.requests)
        pending = sum(r['status'] == 'pending' for r in self.requests)
        totals = {key: sum(r[key] or 0 for r in self.requests) for key in
                  ('input_tokens', 'output_tokens', 'cached_tokens', 'cache_write_tokens')}
        return dict(usd_low=round(sum(r['usd_low'] for r in priced), 10),
                    usd_high=round(sum(r['usd_high'] for r in priced), 10),
                    priced_requests=len(priced), unknown_requests=unknown, pending_requests=pending,
                    request_count=len(self.requests), tokens=totals,
                    token_unknown={key: sum(r['status'] != 'pending' and r[key] is None for r in self.requests) for key in totals},
                    pricing=rates(self.model),
                    requests=self.requests[-100:])
