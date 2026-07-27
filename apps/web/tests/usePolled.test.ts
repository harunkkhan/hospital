import { describe, expect, test } from "bun:test";
import { act, renderHook, waitFor } from "@testing-library/react";

import { usePolled } from "../src/hooks/usePolled";

interface Deferred<T> {
  resolve: (value: T) => void;
  promise: Promise<T>;
}

function makeFetcher<T>(): { fetcher: () => Promise<T>; pending: Deferred<T>[] } {
  const pending: Deferred<T>[] = [];
  const fetcher = (): Promise<T> => {
    let resolve!: (value: T) => void;
    const promise = new Promise<T>((r) => {
      resolve = r;
    });
    pending.push({ resolve, promise });
    return promise;
  };
  return { fetcher, pending };
}

describe("usePolled request ordering", () => {
  // The regression: a shared epoch let an OLDER in-flight response overwrite a
  // NEWER one whenever the slow request settled last. The monotonic id gate
  // must keep the newest issued request authoritative in either settle order.
  test("a slow older response never clobbers a newer one", async () => {
    const { fetcher, pending } = makeFetcher<string>();
    const HUGE = 1_000_000; // effectively disable the interval during the test

    const { result } = renderHook(() => usePolled(fetcher, HUGE));
    expect(pending).toHaveLength(1); // mount issued request #0

    act(() => {
      result.current.refresh(); // issues request #1 (the newer one)
    });
    expect(pending).toHaveLength(2);

    // Newer request settles FIRST...
    await act(async () => {
      pending[1]!.resolve("new");
      await pending[1]!.promise;
    });
    await waitFor(() => expect(result.current.data).toBe("new"));

    // ...then the older request settles LATE and must be ignored.
    await act(async () => {
      pending[0]!.resolve("old");
      await pending[0]!.promise;
    });
    expect(result.current.data).toBe("new");
  });

  test("newest wins even when the older request settles first", async () => {
    const { fetcher, pending } = makeFetcher<string>();
    const { result } = renderHook(() => usePolled(fetcher, 1_000_000));

    act(() => {
      result.current.refresh();
    });

    await act(async () => {
      pending[0]!.resolve("old");
      await pending[0]!.promise;
    });
    await act(async () => {
      pending[1]!.resolve("new");
      await pending[1]!.promise;
    });
    await waitFor(() => expect(result.current.data).toBe("new"));
  });
});
