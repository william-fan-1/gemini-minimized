"""Modal deployment for the Explaining Markets starter.

This is plumbing — you shouldn't need to edit it. It defines a small FastAPI app
and deploys it as a persistent, public web endpoint:

    GET  /    health check
    POST /    receive a signed event, verify, ACK, then predict and submit
              (POST /competition/webhook is kept as an alias of the same handler)

The webhook is served at the root path on purpose: the URL Modal prints on deploy
*is* your webhook URL — paste it into the portal as-is, nothing to append.

Deploy:    uv run modal deploy modal_app.py
Dev/local: uv run modal serve modal_app.py

The webhook handler ACKs first, then predicts. It verifies the signature, returns
200, and spawns `predict_and_submit` — a separate Modal function with its own
container — to run your `predict()` from predict.py and POST the result. Two
clocks:

  * 20 seconds to ACK the delivery. Miss it and the platform retries; repeated
    failures disable your webhook.
  * 5 minutes from that ACK to submit your prediction.

Predicting before the ACK spends the 5-minute budget inside the 20-second one.
Spawning rather than using a background task also means the work doesn't depend
on the web container staying alive.

Deliveries are deduped on the `Webhook-Id` header (the server retries on
4xx/5xx/timeout, so the same event can arrive more than once).

Note: we deliberately do NOT use `from __future__ import annotations` here. The
route handlers are defined inside `web()`, and FastAPI must see the real `Request`
/ `Response` classes (not stringized annotations it can't resolve from this nested
scope) to inject them correctly — otherwise it treats `request` as a query
parameter and rejects every delivery with 422.
"""

import modal

app = modal.App("explaining-markets-gemini-minimized")

image = (
    modal.Image.debian_slim()
    .pip_install("fastapi[standard]", "httpx", "litellm", "pandas", "pydantic", "pyyaml")
    .add_local_python_source("explaining_markets", "predict", "prompt_construction")
    .add_local_dir("prompts", remote_path="/root/prompts")
    .add_local_dir("knowledge", remote_path="/root/knowledge")
)

# Distributed key-value store for idempotency, keyed on the Webhook-Id header.
# Three states:
#
#   "in_flight"   a job is running right now — skip duplicates so you never pay
#                 for the same model call twice
#   "done"        the API accepted a prediction — skip forever
#   absent        never seen, or the last attempt raised — (re)run it
#
# Marking an event done up front would be the bug: a failed prediction would
# look handled. This Dict persists across redeploys, so "done" is durable.
seen_webhooks = modal.Dict.from_name("em-webhook-dedupe-gemini-minimized", create_if_missing=True)
prediction_ledger = modal.Dict.from_name("em-prediction-ledger-gemini-minimized", create_if_missing=True)

# Credentials are read from your local .env at deploy time (see .env.example).
# Prefer Modal's secret store instead? See docs/advanced.md.
secrets = [modal.Secret.from_dotenv(__file__)]


def _claim(webhook_id):
    """Reserve this webhook_id. False means it's already in flight or done.

    `skip_if_exists` makes this an atomic claim, so two containers handling a
    duplicate delivery at the same moment can't both win.
    """
    if not webhook_id:
        return True
    return seen_webhooks.put(webhook_id, "in_flight", skip_if_exists=True)


async def _claim_aio(webhook_id):
    """`_claim` for the async route.

    Modal's blocking interfaces run their own event loop under the hood, so
    calling them from inside an `async def` stalls the loop — the exact problem
    ACKing first is meant to solve. The `.aio` variants are the async-native
    ones; the request path must use these, and only these.
    """
    if not webhook_id:
        return True
    return await seen_webhooks.put.aio(webhook_id, "in_flight", skip_if_exists=True)


def _release(webhook_id, submitted):
    """Mark the claim done on success, or drop it so a redelivery can retry."""
    if not webhook_id:
        return
    if submitted:
        seen_webhooks[webhook_id] = "done"
    else:
        seen_webhooks.pop(webhook_id, None)


def _submission_rows(event, detailed):
    """Build the minimal API payload; metadata is never submission-critical.

    A missing/malformed row falls back to neutral for that focal asset.  This
    also prevents an empty model result from becoming an accepted-looking
    request containing no predictions.
    """
    by_ticker = {}
    for row in detailed or []:
        if not isinstance(row, dict):
            continue
        ticker = row.get("identifier_value")
        value = row.get("predicted_percentile")
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if ticker and 0.0 <= value <= 1.0:
            by_ticker[ticker] = value

    predictions = []
    for asset in event.get("focal_assets", []):
        ticker = asset.get("identifier_value")
        if ticker:
            predictions.append({
                "identifier_value": ticker,
                "predicted_percentile": by_ticker.get(ticker, 0.5),
            })
    return predictions


@app.function(image=image, secrets=secrets, timeout=600, retries=0)
def predict_and_submit(event: dict, webhook_id: str | None = None):
    """Run the model and submit the prediction, off the request path.

    Runs in its own container, so it is unaffected by the web endpoint scaling
    down. The delivery has already been ACKed by the time this starts, which
    means nothing upstream will retry it — the single retry configured on the
    model call in predict.py is the only one you get.
    """
    from explaining_markets.client import submit_predictions
    from explaining_markets.config import Config
    from explaining_markets.event_utils import is_test, neutral_predictions

    submitted = False
    detailed = None
    try:
        if not is_test(event):
            from predict import predict_with_metadata

            # A prediction failure must not become a NON-submission. `predict.py`
            # already falls back to 0.5 on a model error, but the summary fetch
            # ahead of it can raise (`raise_for_status`), and that used to
            # propagate out and skip the submit entirely. Because we ACK 200
            # before predicting, the platform never redelivers — so a dropped
            # event was dropped permanently. The leaderboard then imputes our own
            # mean into the gap, contributing exactly zero. A neutral 0.5 is
            # strictly better than nothing.
            try:
                detailed = predict_with_metadata(event)
            except Exception as exc:
                print(
                    f"[ERROR] predict failed for event {event.get('event_id')}: "
                    f"{type(exc).__name__}: {exc} — submitting neutral 0.5"
                )
                detailed = None

        predictions = (
            neutral_predictions(event)
            if detailed is None
            else _submission_rows(event, detailed)
        )
        submit_predictions(
            event_id=event["event_id"],
            predictions=predictions,
            config=Config.from_env(),
        )
        submitted = True
        # Submission has already succeeded. Everything below is best-effort
        # observability and must not affect the dedupe state or submission.
        if detailed:
            for row in detailed:
                if not isinstance(row, dict) or not row.get("identifier_value"):
                    continue
                ticker = row["identifier_value"]
                try:
                    prediction_ledger[f'{event["event_id"]}:{ticker}'] = {
                        "event_id": event["event_id"],
                        "ticker": ticker,
                        "prompt_version": row.get("prompt_version"),
                        "knowledge_version": row.get("knowledge_version"),
                        "predicted_percentile": row.get("predicted_percentile"),
                        "confidence": row.get("confidence"),
                        "direction": row.get("direction"),
                        "rules_applied": row.get("rules_applied"),
                        "expected_abnormal_return_pct": row.get(
                            "expected_abnormal_return_pct"
                        ),
                        "key_metrics": row.get("key_metrics"),
                        "guidance": row.get("guidance"),
                        "result_quality": row.get("result_quality"),
                        "expectation_gap": row.get("expectation_gap"),
                        "realized_abnormal": None,
                        "realized_percentile": None,
                    }
                except Exception as exc:
                    print(
                        f"[WARN] ledger write failed for {event.get('event_id')}:"
                        f"{ticker}: {type(exc).__name__}: {exc}"
                    )
    except Exception as exc:
        # Log loudly — `modal app logs explaining-markets-starter` finds it.
        print(f"[ERROR] submission failed for event {event.get('event_id')}: {exc}")
    finally:
        _release(webhook_id, submitted)


@app.function(image=image, secrets=secrets)
@modal.asgi_app(label="gemini-minimized")
def web():
    from fastapi import FastAPI, Request, Response

    from explaining_markets import WebhookVerificationError, verify_webhook
    from explaining_markets.config import Config
    from explaining_markets.event_utils import log_deadline

    api = FastAPI(title="Explaining Markets starter")

    @api.get("/")
    def health() -> dict:
        return {"ok": True, "service": "explaining-markets-starter"}

    @api.post("/")
    @api.post("/competition/webhook")  # alias, so an explicit-path URL also works
    async def competition_webhook(request: Request) -> Response:
        config = Config.from_env()

        raw_body = await request.body()  # raw bytes — never request.json()
        try:
            event = verify_webhook(
                raw_body=raw_body,
                headers=request.headers,
                secret=config.webhook_secret,
            )
        except WebhookVerificationError as exc:
            return Response(content=str(exc), status_code=401)

        webhook_id = event.get("id")
        if not await _claim_aio(webhook_id):
            return Response(status_code=200)

        log_deadline(event)
        # Everything slow happens after this 200 goes out. The portal's "Test
        # Webhook" button sends a synthetic TEST event; it takes the same path
        # and submits a neutral prediction (accepted by the API, never scored)
        # so the test exercises your full receive -> submit loop.
        await predict_and_submit.spawn.aio(event, webhook_id)
        return Response(status_code=200)

    return api

@app.function()
def read_prediction_ledger(limit: int | None = None):
    """
    Access rules used by agent in early predictions.
    """
    rows = list(prediction_ledger.values())
    if limit is None:
        limit = len(rows)

    return rows[:limit]
