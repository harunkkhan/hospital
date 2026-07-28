import { describe, expect, test } from "bun:test";
import { fireEvent, render, screen } from "@testing-library/react";

import type { ControlCommand } from "../src/api/types";
import { PlaybackControls, type PlaybackControlsProps } from "../src/components/PlaybackControls";

function renderControls(overrides: Partial<PlaybackControlsProps> = {}): ControlCommand[] {
  const commands: ControlCommand[] = [];
  render(
    <PlaybackControls
      state="paused"
      speed={60}
      simTime={0}
      horizon={604_800_000_000}
      onCommand={(cmd) => commands.push(cmd)}
      bufferLength={10}
      scrubIndex={null}
      onScrub={() => undefined}
      {...overrides}
    />,
  );
  return commands;
}

describe("PlaybackControls", () => {
  test("step is disabled while playing", () => {
    renderControls({ state: "playing" });
    expect((screen.getByRole("button", { name: "Step" }) as HTMLButtonElement).disabled).toBe(true);
  });

  test("step works from paused and sends granularity + count", () => {
    const commands = renderControls({ state: "paused" });
    fireEvent.change(screen.getByLabelText("step granularity"), { target: { value: "event" } });
    fireEvent.click(screen.getByRole("button", { name: "Step" }));
    expect(commands).toEqual([{ action: "step", granularity: "event", count: 1 }]);
  });

  test("play/pause toggle sends the matching command", () => {
    const paused = renderControls({ state: "paused" });
    fireEvent.click(screen.getByRole("button", { name: "Play" }));
    expect(paused).toEqual([{ action: "play" }]);
  });

  test("speed picker sends a speed command with the multiplier", () => {
    const commands = renderControls();
    fireEvent.change(screen.getByLabelText("playback speed"), { target: { value: "240" } });
    expect(commands).toEqual([{ action: "speed", multiplier: 240 }]);
  });

  test("everything but scrub is disabled once finished", () => {
    renderControls({ state: "finished" });
    expect((screen.getByRole("button", { name: "Play" }) as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByRole("button", { name: "Step" }) as HTMLButtonElement).disabled).toBe(true);
  });
});
