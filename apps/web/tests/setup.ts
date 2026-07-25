import { GlobalRegistrator } from "@happy-dom/global-registrator";

GlobalRegistrator.register();

// React 19 act() support for @testing-library/react under bun test.
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

// eslint-disable-next-line import/first -- must run after the registrator.
import { afterEach } from "bun:test";
import { cleanup } from "@testing-library/react";

afterEach(cleanup);
