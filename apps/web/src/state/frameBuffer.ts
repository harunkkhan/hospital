/**
 * Bounded ring of reduced world moments for interpolation and LOCAL scrub.
 *
 * We buffer the post-reduction WorldView (not the raw frame) so every
 * buffered position is a complete, renderable view even when the wire frames
 * are partial deltas. Backward scrub is view-only and bounded by the ring;
 * true past-scrub would need checkpoint replay (doc 07 🟡 §8.9).
 */

import type { StreamFrame } from "../api/types";
import type { WorldView } from "./streamReducer";

export interface BufferedMoment {
  frame: StreamFrame;
  world: WorldView;
}

export class FrameBuffer {
  readonly capacity: number;
  private items: BufferedMoment[] = [];

  constructor(capacity = 600) {
    if (capacity < 1) {
      throw new Error("FrameBuffer capacity must be >= 1");
    }
    this.capacity = capacity;
  }

  get length(): number {
    return this.items.length;
  }

  push(moment: BufferedMoment): void {
    this.items.push(moment);
    if (this.items.length > this.capacity) {
      this.items.splice(0, this.items.length - this.capacity);
    }
  }

  /** index 0 = oldest retained moment. */
  at(index: number): BufferedMoment | undefined {
    return this.items[index];
  }

  latest(): BufferedMoment | undefined {
    return this.items[this.items.length - 1];
  }

  clear(): void {
    this.items = [];
  }
}
