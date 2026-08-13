/**
 * The TypeScript <-> pydantic contract-drift gate (PROGRAMMING_MODEL_SPEC §9).
 *
 * `apps/web/src/api/types.ts` is hand-written, and its own header has said since M2 that a
 * drift check "will compare against" the committed schema artifact. The artifact
 * (`apps/api/schema/openapi.json`) has existed since M2; the check had not, so for two
 * milestones the browser's contract could diverge from the server's with nothing failing.
 *
 * The hole was not hypothetical. Adding `RouteNode.floor` was caught only because the hand
 * edit to `types.ts` happened first and broke a mock fixture — regenerate the schema and
 * leave the TypeScript alone and every gate stays green while the frontend ships a type
 * that is missing a field the server sends. The reverse is worse: a field *removed* server
 * side leaves the frontend promising data that never arrives.
 *
 * The interfaces are read with the TypeScript parser rather than a regex, so the check is
 * exact about optionality and cannot be fooled by formatting.
 *
 * Scope, deliberately narrow: **field names and required-ness**, for the models the web app
 * actually consumes. Types are not compared — `tsc` already governs whether the code uses
 * them consistently, and the wire flattens pydantic newtypes (every typed id is a bare
 * string), so a structural type match would be re-encoding that flattening rule here in a
 * second place. Names and requiredness are where silent drift actually happens.
 */

import { readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "bun:test";
import ts from "typescript";

const REPO_ROOT = join(import.meta.dir, "..", "..", "..");
const SCHEMA_PATH = join(REPO_ROOT, "apps", "api", "schema", "openapi.json");
const TYPES_PATH = join(REPO_ROOT, "apps", "web", "src", "api", "types.ts");

/**
 * Wire models the web app hand-types and the server also publishes.
 *
 * An explicit list, not "everything in the schema": the API publishes request bodies and
 * internal shapes the console never touches, and asserting those would fail for models the
 * frontend has no business knowing about. Adding a model here is the deliberate act of
 * saying "the browser depends on this shape".
 */
const CHECKED = [
  "RouteNode",
  "RouteEdge",
  "RouteGraph",
  "Bay",
  "Zone",
  "FloorLayout",
] as const;

// Not listed, and worth saying why: `StaffMember` and `StaffState` are core types the API
// never publishes — they reach the browser only as the flattened fields of a stream frame —
// so neither side declares them and asserting them would fail for a shape that correctly
// does not exist. The list is the intersection of "the server publishes it" and "the browser
// hand-types it", which is exactly the surface where drift can hurt.

interface Members {
  required: Set<string>;
  optional: Set<string>;
}

function tsInterfaces(): Map<string, Members> {
  const source = ts.createSourceFile(
    TYPES_PATH,
    readFileSync(TYPES_PATH, "utf8"),
    ts.ScriptTarget.ES2022,
    true,
  );
  const out = new Map<string, Members>();
  source.forEachChild((node) => {
    if (!ts.isInterfaceDeclaration(node)) {
      return;
    }
    const required = new Set<string>();
    const optional = new Set<string>();
    for (const member of node.members) {
      if (!ts.isPropertySignature(member) || member.name === undefined) {
        continue;
      }
      const name = member.name.getText(source);
      (member.questionToken === undefined ? required : optional).add(name);
    }
    out.set(node.name.text, { required, optional });
  });
  return out;
}

interface SchemaModel {
  required: Set<string>;
  all: Set<string>;
}

function schemaModels(): Map<string, SchemaModel> {
  const doc = JSON.parse(readFileSync(SCHEMA_PATH, "utf8")) as {
    components: { schemas: Record<string, { properties?: Record<string, unknown>; required?: string[] }> };
  };
  const out = new Map<string, SchemaModel>();
  for (const [name, model] of Object.entries(doc.components.schemas)) {
    out.set(name, {
      all: new Set(Object.keys(model.properties ?? {})),
      // A pydantic field with a default is not "required" on the wire, but the server
      // always sends it — so the browser may type it as required. Requiredness is
      // therefore asserted in one direction only; see the test below.
      required: new Set(model.required ?? []),
    });
  }
  return out;
}

describe("TypeScript <-> pydantic contract drift", () => {
  const interfaces = tsInterfaces();
  const schemas = schemaModels();

  it("finds every checked model on both sides", () => {
    for (const name of CHECKED) {
      expect(schemas.has(name), `${name} missing from openapi.json`).toBe(true);
      expect(interfaces.has(name), `${name} missing from types.ts`).toBe(true);
    }
  });

  it.each([...CHECKED])("%s declares exactly the schema's fields", (name) => {
    const iface = interfaces.get(name);
    const schema = schemas.get(name);
    if (iface === undefined || schema === undefined) {
      throw new Error(`${name} is absent on one side — see the test above`);
    }
    const declared = new Set([...iface.required, ...iface.optional]);

    // A field the server sends that the browser does not know about. This is the failure
    // that adding `RouteNode.floor` would have caused had the TypeScript not been edited.
    const missing = [...schema.all].filter((f) => !declared.has(f));
    expect(missing, `${name}: types.ts is missing server field(s)`).toEqual([]);

    // A field the browser expects that the server does not send — worse, because the code
    // reads it and gets undefined at runtime with no type error.
    const extra = [...declared].filter((f) => !schema.all.has(f));
    expect(extra, `${name}: types.ts declares field(s) the server does not send`).toEqual([]);
  });

  it.each([...CHECKED])("%s never types an absent field as required", (name) => {
    const iface = interfaces.get(name);
    const schema = schemas.get(name);
    if (iface === undefined || schema === undefined) {
      throw new Error(`${name} is absent on one side — see the test above`);
    }
    // One direction only. A defaulted pydantic field is omitted from `required` in the
    // schema, yet the server always serializes it, so typing it non-optional in TS is
    // correct and common (`RouteNode.floor` is exactly that). What must never happen is a
    // TS field marked required that the schema does not carry at all — that is a promise
    // about bytes nobody sends, and the check above already forbids it.
    for (const field of iface.required) {
      expect(schema.all.has(field), `${name}.${field} is required in TS but absent server-side`).toBe(true);
    }
  });
});
