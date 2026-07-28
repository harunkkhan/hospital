/**
 * SSE stream client: GET /runs/{id}/stream with `Accept: text/event-stream`.
 *
 * Reconnect protocol (doc 07 nuances §7.3): there is no server-side replay
 * buffer — on reconnect the server sends one fresh `snapshot` frame, then
 * deltas. So recovery is always "reconnect and re-snapshot", never "catch up
 * from seq". EventSource's native retry covers transient drops; we add capped
 * exponential backoff on hard errors and surface status transitions so the UI
 * can distinguish "paused" from "socket stalled".
 */

import type { StreamCallbacks, StreamHandle } from "./client";
import type { StreamFrame } from "./types";

const MAX_BACKOFF_MS = 10_000;
const BASE_BACKOFF_MS = 500;

export function openSseStream(url: string, callbacks: StreamCallbacks): StreamHandle {
  let source: EventSource | null = null;
  let closed = false;
  let attempts = 0;
  let retryTimer: ReturnType<typeof setTimeout> | null = null;

  const connect = (): void => {
    if (closed) {
      return;
    }
    callbacks.onStatus?.(attempts === 0 ? "connecting" : "reconnecting");
    source = new EventSource(url);

    source.onopen = () => {
      attempts = 0;
      callbacks.onStatus?.("open");
    };

    source.onmessage = (msg: MessageEvent<string>) => {
      let frame: StreamFrame;
      try {
        frame = JSON.parse(msg.data) as StreamFrame;
      } catch {
        return; // heartbeat/comment payloads are not frames
      }
      callbacks.onFrame(frame);
    };

    source.onerror = () => {
      if (closed) {
        return;
      }
      // EventSource retries CONNECTING states itself; a CLOSED source needs us.
      if (source?.readyState === EventSource.CLOSED) {
        source.close();
        source = null;
        attempts += 1;
        const backoff = Math.min(MAX_BACKOFF_MS, BASE_BACKOFF_MS * 2 ** (attempts - 1));
        callbacks.onStatus?.("reconnecting");
        retryTimer = setTimeout(connect, backoff);
      }
    };
  };

  connect();

  return {
    close() {
      closed = true;
      if (retryTimer !== null) {
        clearTimeout(retryTimer);
      }
      source?.close();
      source = null;
      callbacks.onStatus?.("closed");
    },
  };
}
