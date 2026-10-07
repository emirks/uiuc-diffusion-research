#!/usr/bin/env python3
"""Build prompts.jsonl for the collapse-probe experiment (base LTX-2, no adapter)."""
import json, os

GRID = "store/gens/005_base_cond/04_neutral_v3__dai/grid.jsonl"
OUT_DIR = "misc/2026-09-08_collapse_probe/prompts"

# --- load grid, dedupe by endpoint ---
seen = {}
for line in open(GRID):
    r = json.loads(line)
    seen.setdefault(r["endpoint"], r)

# selection order (30), max 2 per class
SELECT = ["wireframe_7","wireframe_2","portal_1","portal_10","explosion_1","explosion_0",
"plasma_explosion_2","point_cloud_1","point_cloud_0","monstrosity_0","water_element_3",
"water_element_4","northern_lights_0","fire_element_2","fire_element_3","color_rain_2",
"color_rain_0","earth_element_6","earth_element_4","train_rush_1","train_rush_0",
"money_rain_3","glitch_1","acid_0","luminous_gaze_3","saint_glow_3","live_concert_4",
"animalization_0","shadow_3","gas_transformation_1"]

# HIGH tier: (mechanism_family, change_clause, end_caption). One unique mechanism family each.
HIGH = {
"wireframe_7": ("crumble-to-dust",
  "The rocks and cacti crumble into fine grey dust that gusts sideways out of the frame, and as the last grains clear a completely different place stands in their place.",
  "A vibrant coral reef spreads across a sandy seabed, orange and purple corals swaying gently while small silver fish dart between them and shafts of sunlight filter down through clear blue water."),
"wireframe_2": ("paper-fold",
  "The flat image folds along sharp creases like a sheet of paper, tucking the shirt and pavement away, then unfolds face-out onto an entirely new location.",
  "A cozy bookshop interior glows under warm lamplight, tall wooden shelves crammed with colourful spines lining a narrow aisle, a worn red armchair in the corner beside a small round reading table."),
"portal_1": ("whirlpool-vortex",
  "The branch and the little figure spiral inward into a slow whirlpool of colour that winds tighter and then unwinds outward, opening onto a different world.",
  "A candy-coloured cartoon meadow rolls toward the horizon, oversized striped mushrooms and swirled lollipop trees dotting rounded green hills under a bright turquoise sky flecked with fluffy white clouds."),
"portal_10": ("neon-streak-reshape",
  "Streaks of neon light stretch across the alley into long glowing lines that snap and rearrange themselves into the bright outlines of a new scene.",
  "A neon-lit cartoon arcade stretches back in rows of glowing game cabinets, their screens flashing pink and blue, a chequered floor reflecting the colours beneath a low starry ceiling."),
"explosion_1": ("dive-through-window",
  "The camera surges forward and dives straight through a lit apartment window across the street, punching cleanly through the glass into the room waiting beyond.",
  "A warm living room opens up with a crackling fireplace, two overstuffed sofas facing a low wooden coffee table, framed pictures on the mantel and soft golden lamplight spilling across a patterned rug."),
"explosion_0": ("ocean-wave-crash",
  "A towering ocean wave sweeps in from the side and crashes across the whole frame, then drags back and pulls the street away with it.",
  "A wide sandy beach curves along a turquoise bay at low tide, gentle foam-edged waves rolling in, a lone wooden lifeguard tower standing against a bright sky streaked with thin white clouds."),
"plasma_explosion_2": ("ground-shatter-glass",
  "The street shatters beneath the car into countless glass shards that lift, spin, and reassemble in mid-air into a different setting.",
  "A sleek modern art gallery stretches down a bright white hall, large abstract canvases spaced along the walls, a polished concrete floor reflecting recessed ceiling lights and a single sculptural bench at the centre."),
"point_cloud_1": ("sandstorm-sweep",
  "A wall of golden sand sweeps across the frame from the right, blotting out the car entirely, then thins and clears to reveal somewhere new.",
  "A vast desert of rippled orange dunes rolls to the horizon under a deep evening sky, long shadows raking across the sand and a lone caravan of camels tracing a distant ridge."),
"point_cloud_0": ("train-wipe",
  "A train rushes past across the frame in a blur of speeding carriages, and as the last car clears it wipes the platform away behind it.",
  "A quiet mountain railway halt sits among tall pines, a small wooden station house with hanging flower baskets beside a single empty track, snow-dusted peaks rising sharply in the crisp morning light."),
"monstrosity_0": ("fog-roll-in",
  "Thick white fog rolls in low across the white flowers and swallows the whole scene, then lifts slowly to uncover a different landscape.",
  "A misty green rice terrace steps down a steep valley in wide curving tiers, thin channels of water catching the pale sky, a farmer in a conical hat wading through the flooded paddies."),
"water_element_3": ("blizzard-whiteout",
  "A sudden blizzard whips across the slope and whites out the entire frame in swirling snow, then settles and thins to reveal a new place.",
  "A warm alpine chalet interior glows with firelight, exposed timber beams overhead, a stone hearth stacked with logs and thick wool blankets draped over a leather couch while steam rises from a mug."),
"water_element_4": ("mirror-crack",
  "The view cracks like a sheet of mirror glass, the fractured pieces tilting and falling away one by one to expose a different location behind them.",
  "A bustling covered market fills with colour, stalls heaped with ripe fruit and hanging lanterns lining a narrow walkway, shoppers weaving past under strings of paper flags in warm afternoon light."),
"northern_lights_0": ("aurora-curtain",
  "Shimmering curtains of green aurora sweep down and draw across the frame like heavy drapes, then part to either side onto somewhere completely different.",
  "A dense tropical jungle steams in humid daylight, broad dripping leaves and hanging vines crowding a narrow trail, a slender waterfall tumbling into a clear pool ringed by mossy grey boulders."),
"fire_element_2": ("flame-wall",
  "A curtain of fire sweeps across the black sand from the left and roars over the frame, then burns down and clears to reveal a new scene.",
  "A snow-covered village square rests under soft grey light, timber-framed houses with steep white roofs ringing a frozen fountain, warm lamplight in the windows and a fir tree strung with tiny glowing lights."),
"fire_element_3": ("bird-swarm",
  "A dense swarm of black birds sweeps up from the field and blots out the whole frame, then scatters apart over a different place.",
  "A grand old library reading room stretches upward two storeys, brass-railed balconies of leather-bound books, green banker's lamps glowing along long oak tables and tall arched windows admitting soft afternoon light."),
"color_rain_2": ("ink-flood",
  "A wash of dark ink floods in from every edge of the frame and drowns the desert, then drains rapidly away to leave a new scene behind.",
  "A rain-slicked city street glistens at night under neon signs, their pink and blue glow smearing across the wet asphalt, a lone yellow taxi idling at a corner beside a steaming vendor's cart."),
"color_rain_0": ("freeze-and-shatter",
  "A sheet of frost races across the kitchen and freezes everything to glittering ice, which then cracks and falls away to reveal a new scene.",
  "A quiet snowbound forest clearing rests under pale light, tall firs heavy with snow ringing a frozen pond, a small log cabin with a smoking chimney and soft drifts curving between the trees."),
"earth_element_6": ("earth-splits",
  "The canyon floor splits open along a jagged crack and the camera plunges down through the widening chasm into the space below.",
  "A vast crystal cavern glows with soft blue light, enormous translucent columns rising from a mirror-still underground lake, glittering veins threading the dark rock walls and a faint mist drifting over the water."),
"earth_element_4": ("puddle-reflection",
  "The camera tips down and pushes into a rain puddle on the street, sinking through its rippling reflection and rising back up somewhere new.",
  "A quiet Venetian canal curves between faded ochre buildings, a narrow gondola gliding past shuttered windows and small arched bridges while morning light glints off the calm green water."),
"train_rush_1": ("light-bloom",
  "A swelling bloom of white light rises from the altar and floods outward until it engulfs the frame, then subsides to reveal a different place.",
  "A sunlit wildflower meadow rolls gently uphill, waves of red poppies and blue cornflowers nodding in the breeze, a single gnarled oak on the crest and butterflies drifting under a bright open sky."),
"train_rush_0": ("kaleidoscope-fracture",
  "The frame fractures into a spinning kaleidoscope of repeating wedges that whirl and multiply, then resolve and settle into a single new scene.",
  "A calm lakeside dock reaches out over still water at dawn, a small rowboat tied at its end, pine-covered hills mirrored on the glassy surface and thin mist curling above the shallows."),
"money_rain_3": ("wax-slump",
  "The whole scene softens and slumps downward as if moulded from warm wax, sliding out of the frame and re-forming smoothly into a different place.",
  "A busy fish market spreads under a high steel roof, crates of glistening catch on crushed ice, vendors in rubber aprons calling out and bright overhead lights reflecting off the wet tiled floor."),
"glitch_1": ("cubes-reassemble",
  "The picture breaks apart into thousands of floating cubes that drift and tumble through the air, then click together into a completely different scene.",
  "A retro diner glows under warm neon, red vinyl booths along a chrome counter, a chequerboard floor and tall milkshake glasses catching the light while a jukebox glows softly in the corner."),
"acid_0": ("smoke-plume",
  "A billowing plume of pale smoke rolls across the deck and engulfs the whole frame, then thins and clears over somewhere entirely different.",
  "A dense bamboo forest rises in tall green stalks, soft light slanting between them onto a mossy stone path, a small wooden shrine half-hidden among the leaves and gentle mist hanging in the still air."),
"luminous_gaze_3": ("vines-overgrow",
  "Green vines and broad leaves rush across the frame from the edges and knit together over everything, then peel back to uncover a new place.",
  "A grand greenhouse conservatory arches overhead in white iron and glass, lush palms and ferns crowding gravel paths, a tiered stone fountain trickling at the centre under bright diffuse daylight."),
"saint_glow_3": ("petal-swarm",
  "A gust lifts a storm of pink blossom petals across the sidewalk until they fill the frame, then they scatter away to reveal a different setting.",
  "A tranquil Japanese garden opens around a still koi pond, an arched red bridge crossing to a mossy islet, carefully raked gravel and a stone lantern set beneath a blooming cherry tree."),
"live_concert_4": ("rain-sheet",
  "A heavy sheet of rain sweeps down the frame in a shimmering wall of water, then clears from top to bottom onto a completely different place.",
  "A cozy mountain cabin porch looks out over a pine valley at dusk, a wooden rocking chair beside a lit lantern, string lights glowing along the eaves and distant peaks receding into blue shadow."),
"animalization_0": ("wallpaper-peel",
  "The blue backdrop peels away from one corner like a sheet of wallpaper, rolling steadily across the frame to expose a different scene beneath it.",
  "A sunlit Tuscan hillside rolls in golden waves, rows of dark green cypress trees lining a winding dirt road, a terracotta-roofed farmhouse in the distance under a wide, warm afternoon sky."),
"shadow_3": ("shadow-swallow",
  "A deep shadow spreads across the blue background and swallows the frame into darkness, then draws back like a receding tide to reveal a new place.",
  "A busy night food street glows with hanging red lanterns, steam rising from rows of open-air stalls, crowds squeezing past bright signage and plastic stools clustered along the narrow lane."),
"gas_transformation_1": ("ripple-distortion",
  "The whole frame ripples and warps as if it were the surface of a pond struck by a stone, the wobbling distortion settling into a different scene.",
  "A snow-globe alpine ski village nestles in a white valley, chalets with glowing windows lining a snowy street, a chairlift climbing toward pale peaks and skiers gliding past under gently falling snow."),
}

# LOW tier controls (10). Same place/subject, small in-place change.
LOW_ORDER = ["wireframe_7","explosion_0","point_cloud_0","monstrosity_0","water_element_3",
             "fire_element_2","color_rain_0","earth_element_4","glitch_1","animalization_0"]
LOW = {
"wireframe_7": (
  "The daylight warms gently and a soft highlight blooms on the green glass bottle, while the camera eases slowly closer and a few dust motes drift through the air.",
  "The same green serum bottle sits among the cacti on the rocks, now catching a warm soft highlight on its glass as the light turns golden and a few dust motes drift slowly through the air."),
"explosion_0": (
  "The warm afternoon light deepens toward a golden hue and the shadows stretch longer across the crosswalk, while a gentle breeze lifts the woman's skirt and hair.",
  "The same woman in her grey top and beige skirt crosses the crosswalk between the parked cars, the afternoon light now a deeper gold with long soft shadows reaching across the street."),
"point_cloud_0": (
  "The bright platform light softens to a warmer tone and the camera pushes in slightly, while the reflections on the subway car windows shift and glimmer gently.",
  "The same person in the white cap sits on the bench in front of the blue and white subway car, the harsh light now warmer and softer while the camera frames them a little more closely."),
"monstrosity_0": (
  "The overcast light warms gently toward gold and a soft mist begins to drift low across the white flowers, while the camera eases almost imperceptibly closer.",
  "The same young man in his long grey scarf stands on the rock among the white flowers, the snow-capped mountains behind him now touched with warm golden light as thin mist drifts low across the field."),
"water_element_3": (
  "The clear blue sky warms toward a pale afternoon gold and light snowflakes begin to drift down, while a gentle breeze stirs the fluffy trim of the woman's jacket.",
  "The same blonde woman in her white fur-trimmed jacket stands in the snowy mountain landscape, the sky now a warm pale gold as fine snowflakes drift down and a light breeze lifts her hair."),
"fire_element_2": (
  "The overcast sky brightens softly and a warm rim of sunlight edges the clouds, while a light wind lifts the hem of the red coat and ruffles the woman's hair.",
  "The same woman in her long red coat stands on the black sand beach, the cloudy sky now brightening with a warm glow at its edges as a gentle sea breeze stirs her hair and coat."),
"color_rain_0": (
  "The warm kitchen light brightens gently and a soft push-in eases the camera a little closer, while steam begins to curl upward from a mug on the counter.",
  "The same woman stands in her warm-lit kitchen holding the white telephone receiver, framed slightly closer now, with gentle steam rising from a mug on the counter as the afternoon light grows warmer."),
"earth_element_4": (
  "The daylight shifts toward a warm late-afternoon glow and the shadows lengthen slightly, while the camera pushes in gently and a few leaves drift down past the parked SUV.",
  "The same young man in his blue varsity jacket stands in front of the white SUV on the city street, now lit by warm late-afternoon sun with long soft shadows and a few leaves drifting down."),
"glitch_1": (
  "The soft studio light dims slightly to a cooler blue and a faint warm glow rises from the television screen, while the camera drifts a little closer.",
  "The same dark-haired woman rests her chin on her hands atop the vintage television, the studio now lit in a cooler blue with a soft warm glow spilling from the screen showing her face."),
"animalization_0": (
  "The studio lights slowly warm from cool white to a soft golden tone and the blue backdrop deepens a shade, while a gentle breeze stirs the woman's hair.",
  "The same woman in her red bomber jacket stands against the blue studio backdrop, now bathed in warm golden light that softens the shadows on her face while her hair lifts gently in a light breeze."),
}

def clip_path(ep):
    return f"data/processed/transitions_std121/{seen[ep]['endpoint_class']}/{ep}.mp4"

rows = []
for i, ep in enumerate(SELECT, start=1):
    r = seen[ep]
    sc = r["prompt"]
    mech, cc, ec = HIGH[ep]
    rows.append({
        "prompt_id": f"P{i:02d}H", "endpoint": ep, "endpoint_class": r["endpoint_class"],
        "endpoint_source": r["endpoint_source"], "tier": "high",
        "start_caption": sc, "change_clause": cc, "end_caption": ec,
        "mechanism_family": mech,
        "full_prompt": f"{sc} {cc} {ec}", "neutral_prompt": f"{sc} {ec}",
        "clip_path": clip_path(ep),
    })

# LOW rows keep the same clip index number as their HIGH counterpart
idx = {ep: i for i, ep in enumerate(SELECT, start=1)}
for ep in LOW_ORDER:
    r = seen[ep]
    sc = r["prompt"]
    cc, ec = LOW[ep]
    i = idx[ep]
    rows.append({
        "prompt_id": f"P{i:02d}L", "endpoint": ep, "endpoint_class": r["endpoint_class"],
        "endpoint_source": r["endpoint_source"], "tier": "low",
        "start_caption": sc, "change_clause": cc, "end_caption": ec,
        "mechanism_family": "in-place",
        "full_prompt": f"{sc} {cc} {ec}", "neutral_prompt": f"{sc} {ec}",
        "clip_path": clip_path(ep),
    })

with open(os.path.join(OUT_DIR, "prompts.jsonl"), "w") as f:
    for row in rows:
        f.write(json.dumps(row) + "\n")
print("wrote", len(rows), "rows")
