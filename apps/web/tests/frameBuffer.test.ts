import { describe, expect, test } from "bun:test";

import { FrameBuffer, type BufferedMoment } from "../src/state/frameBuffer";
import { initialWorld } from "../src/state/streamReducer";
import type { StreamFrame } from "../src/api/types";

function moment(seq: number): BufferedMoment {
  const frame = { seq } as StreamFrame;
  return { frame, world: { ...initialWorld(), seq } };
}

describe("FrameBuffer", () => {
  test("keeps at most capacity moments, dropping the oldest", () => {
    const buffer = new FrameBuffer(3);
    for (let i = 0; i < 5; i += 1) {
      buffer.push(moment(i));
    }
    expect(buffer.length).toBe(3);
    expect(buffer.at(0)?.frame.seq).toBe(2);
    expect(buffer.latest()?.frame.seq).toBe(4);
  });

  test("at() addresses oldest-first and out-of-range is undefined", () => {
    const buffer = new FrameBuffer(10);
    buffer.push(moment(7));
    expect(buffer.at(0)?.frame.seq).toBe(7);
    expect(buffer.at(1)).toBeUndefined();
  });

  test("clear() empties the ring", () => {
    const buffer = new FrameBuffer(10);
    buffer.push(moment(1));
    buffer.clear();
    expect(buffer.length).toBe(0);
    expect(buffer.latest()).toBeUndefined();
  });

  test("rejects a non-positive capacity", () => {
    expect(() => new FrameBuffer(0)).toThrow();
  });
});
