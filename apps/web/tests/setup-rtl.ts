/** Preload step 2: React Testing Library auto-cleanup (after the DOM exists). */

import { afterEach } from "bun:test";
import { cleanup } from "@testing-library/react";

afterEach(cleanup);
