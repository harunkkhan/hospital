/**
 * Preload step 1: install the happy-dom globals BEFORE anything imports
 * @testing-library/* (its `screen` binds document.body at module init).
 * Step 2 lives in setup-rtl.ts — bunfig preloads them in order.
 */

import { GlobalRegistrator } from "@happy-dom/global-registrator";

GlobalRegistrator.register();

// React 19 act() support for @testing-library/react under bun test.
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
