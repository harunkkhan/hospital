/**
 * The style packs: two design directions over one geometry derivation.
 *
 * A pack is DATA ONLY — colours and opacities. Nothing in `scene.ts` knows which pack it is
 * drawing, so a third direction is a new entry in this table rather than a fork of the
 * builders. That seam is the whole point of keeping the type: the prototype carried four
 * packs and dropping two of them cost one array entry each.
 *
 * **Blueprint** is the default and the look this view is tuned for: hairline ink on
 * near-white, structure over surface, so the corridor grid survives every zoom. **Clinical
 * Dark** is the console's own palette — its ramps are imported from `render/colors.ts`
 * rather than restated, because a status colour that disagrees between the 2D map, the
 * Legend and the 3D floor is a defect, not a style.
 */

import type { BayStatus, EsiAcuity, ZoneType } from "../api/types";
import {
  BAY_STATUS_COLORS,
  ESI_COLORS,
  LABEL_COLOR,
  SELECTION_COLOR,
  STAFF_DOT_COLOR,
} from "../render/colors";
import type { RoomKind } from "./derive";

/** Every label, on canvas and in this view's own chrome. */
export const LABEL_FONT = '"Times New Roman", Times, serif';

/** Zone hues are keyed by a bay's zone type, or by the kind of a non-clinical room. */
export type TintKey = ZoneType | RoomKind;

export interface StylePack {
  readonly id: string;
  readonly name: string;
  readonly tagline: string;
  /** Scene clear colour. */
  readonly bg: string;
  readonly plateFill: string;
  readonly deptFill: string;
  readonly corridorFill: string;
  readonly roomFill: string;
  readonly ink: string;
  readonly inkStrong: string;
  readonly inkSoft: string;
  readonly label: string;
  /**
   * Both shipped packs draw walls as hairlines on an unlit scene. The prototype also carried
   * a solid-poché mode and a lit massing mode; neither survived the cut, and the fields that
   * drove them are gone with them rather than sitting here as branches nothing takes.
   */
  readonly wall: { readonly edgeOp: number };
  readonly room: { readonly fillOp: number };
  readonly slabOp: number;
  readonly gridOp: number;
  readonly statusOp: number;
  readonly furnitureOp: number;
  readonly furnitureEdgeOp: number;
  readonly status: Readonly<Record<BayStatus, string>>;
  readonly esi: Readonly<Record<EsiAcuity, string>>;
  readonly staff: string;
  /** The selected bay's outline — it must read against both the floor and every status. */
  readonly selection: string;
  readonly zoneTint: Partial<Readonly<Record<TintKey, string>>>;
  readonly zoneTintOp: number;
}

const BLUEPRINT: StylePack = {
  id: "blueprint",
  name: "Blueprint",
  tagline: "Hairline ink on near-white. Structure over surface; status colour is muted by design.",
  bg: "#e6eaf0",
  plateFill: "#dfe4eb",
  deptFill: "#fcfcfd",
  corridorFill: "#e9edf2",
  roomFill: "#ffffff",
  ink: "#5a5a5a",
  inkStrong: "#161616",
  inkSoft: "#b3b8bf",
  label: "#4d4d4d",
  wall: { edgeOp: 0.72 },
  room: { fillOp: 0.5 },
  slabOp: 0.85,
  gridOp: 0.1,
  statusOp: 0.3,
  furnitureOp: 0.22,
  furnitureEdgeOp: 0.5,
  // Ink-weight equivalents of the console ramp: on paper the saturated dark-surface steps
  // read as fluorescent, so each is taken down to something a plotter could print.
  status: { free: "#12854a", occupied: "#2f6fd0", cleaning: "#b07908", closed: "#b3352f" },
  esi: { 1: "#b3352f", 2: "#c2611f", 3: "#a97a09", 4: "#12854a", 5: "#8a8a8a" },
  // Slate rather than near-black: a body at this scale is a silhouette, and pure ink on
  // near-white reads as a bollard. Still clear of every acuity hue.
  staff: "#4a5464",
  selection: "#161616",
  zoneTint: {
    fast_track: "#7fb3d5",
    general: "#a9c3d9",
    observation: "#b9c9b0",
    resus_trauma: "#dba9a3",
    triage: "#c8b8d9",
    imaging: "#b6c8cc",
    lab: "#cfc6ac",
    support: "#c8ccd2",
  },
  zoneTintOp: 0.16,
};

const CLINICAL_DARK: StylePack = {
  id: "clinical",
  name: "Clinical Dark",
  tagline: "The console's own dark-first palette in 3D. Status reads hardest; the building recedes to ink.",
  bg: "#0d0d0d",
  plateFill: "#131417",
  deptFill: "#191a1d",
  corridorFill: "#222327",
  roomFill: "#1e1f23",
  ink: "#54534f",
  inkStrong: "#98968f",
  inkSoft: "#2f2f2d",
  label: LABEL_COLOR,
  wall: { edgeOp: 0.7 },
  room: { fillOp: 0.9 },
  slabOp: 0.95,
  gridOp: 0.07,
  statusOp: 0.42,
  furnitureOp: 0.5,
  furnitureEdgeOp: 0.7,
  status: BAY_STATUS_COLORS,
  esi: ESI_COLORS,
  staff: STAFF_DOT_COLOR,
  selection: SELECTION_COLOR,
  zoneTint: {
    fast_track: "#2b6f8f",
    general: "#2f4d6b",
    observation: "#37613f",
    resus_trauma: "#7a2f2f",
    triage: "#4d3a6b",
    imaging: "#2f5f63",
    lab: "#5f5330",
    support: "#3a3b40",
  },
  zoneTintOp: 0.4,
};

export const STYLE_PACKS: readonly StylePack[] = [BLUEPRINT, CLINICAL_DARK];

/** The look this view is tuned for; anything unrecognised falls back to it. */
export const DEFAULT_STYLE_ID = BLUEPRINT.id;

export function stylePackById(id: string): StylePack {
  return STYLE_PACKS.find((pack) => pack.id === id) ?? BLUEPRINT;
}
