# Collapse-probe prompt set — base LTX-2 (no adapter)

Built 2026-09-08. Deliverable = input to the R1/R2/R3 collapse experiment: 30 start clips (one-sided classes, so the start carries no scene-change bias), each with a HIGH-drama three-part prompt describing a transition into a DIFFERENT scene; 10 of the 30 also get a LOW-drama in-place control.

- `prompts.jsonl` — 40 rows, machine-readable (fields: prompt_id, endpoint, endpoint_class, endpoint_source, tier, start_caption, change_clause, end_caption, mechanism_family, full_prompt, neutral_prompt, clip_path).
- `full_prompt` = start_caption + change_clause + end_caption (the transition condition).
- `neutral_prompt` = start_caption + end_caption (anchors-only condition, R3 — names the two scenes, no mechanism).
- Banned words (would bias toward the collapse family): dissolve, crossfade, fade, cut, lerp. None appear.

## Selection rationale

From 61 eligible endpoints (sided==one, source in {heldin_test, heldout, heldin_train}, clip on disk) we picked 30, at most 2 per class, maximising SETTING diversity. Captions were confirmed against the real first frame of each clip (see `startclips_montage.png`).

**Class distribution (HIGH, 21 classes):** color_rain×2, earth_element×2, explosion×2, fire_element×2, point_cloud×2, portal×2, train_rush×2, water_element×2, wireframe×2, acid×1, animalization×1, gas_transformation×1, glitch×1, live_concert×1, luminous_gaze×1, money_rain×1, monstrosity×1, northern_lights×1, plasma_explosion×1, saint_glow×1, shadow×1.

**Diversity axes covered (by primary setting):**
- object: wireframe_7, super_fast_run_10
- cartoon/fantasy subject: portal_1, portal_10
- vehicle: explosion_1, plasma_explosion_2, point_cloud_1
- landscape: monstrosity_0, water_element_3, northern_lights_0, fire_element_2, color_rain_2, earth_element_6
- cityscape: explosion_0, water_element_4, earth_element_4, train_rush_0, acid_0, luminous_gaze_3, saint_glow_3, live_concert_4
- interior: color_rain_0, train_rush_1, money_rain_3
- interior/transit: point_cloud_0
- interior/studio: glitch_1
- field: fire_element_3, gas_transformation_1
- studio portrait: animalization_0, shadow_3

**Lighting / time:** night — northern_lights_0, acid_0, train_rush_0, live_concert_4, portal_10; sunset — fire_element_3; dim interior — train_rush_1; the rest daylight/studio. **Non-people subjects** (rare in this effects dataset): wireframe_7 (bottle), super_fast_run_10 (t-shirt), portal_1/portal_10 (cartoon characters). No true animal start clip exists among eligible one-sided endpoints — noted as a coverage gap.

**Camera motion** could not be judged reliably from a single first frame; most captions describe a subject standing/sitting (static framing). Not used as a hard selection axis.

## LOW-tier controls (10)

In-place effect on the SAME scene (light/colour shift, gentle push-in, small element), spread across 10 distinct classes and setting types: wireframe_7, explosion_0, point_cloud_0, monstrosity_0, water_element_3, fire_element_2, color_rain_0, earth_element_4, glitch_1, animalization_0.

## Prompt listing

### HIGH tier (30)

### P01H  [wireframe_7 · wireframe · heldin_test · HIGH · mechanism: crumble-to-dust]
- **start_caption** (verbatim): A green serum bottle sits among cacti on rocks.
- **change_clause**: The rocks and cacti crumble into fine grey dust that gusts sideways out of the frame, and as the last grains clear a completely different place stands in their place.
- **end_caption**: A vibrant coral reef spreads across a sandy seabed, orange and purple corals swaying gently while small silver fish dart between them and shafts of sunlight filter down through clear blue water.

### P02H  [super_fast_run_10 · wireframe · heldin_train · HIGH · mechanism: paper-fold]
- **start_caption** (verbatim): A man in sunglasses, a dark green hoodie and white cap stands in a bright industrial kitchen.
- **change_clause**: The kitchen folds along sharp creases like a sheet of paper, tucking the man and the steel counters away, then unfolds face-out onto an entirely new location.
- **end_caption**: A cozy bookshop interior glows under warm lamplight, tall wooden shelves crammed with colourful spines lining a narrow aisle, a worn red armchair in the corner beside a small round reading table.

### P03H  [portal_1 · portal · heldin_test · HIGH · mechanism: whirlpool-vortex]
- **start_caption** (verbatim): A small, green figure with a glowing yellow mushroom cap and large white eyes stands on a textured tree branch, wearing a dark double-breasted jacket while holding an orange sphere and pointing upward against a blue and green gradient background.
- **change_clause**: The branch and the little figure spiral inward into a slow whirlpool of colour that winds tighter and then unwinds outward, opening onto a different world.
- **end_caption**: A candy-coloured cartoon meadow rolls toward the horizon, oversized striped mushrooms and swirled lollipop trees dotting rounded green hills under a bright turquoise sky flecked with fluffy white clouds.

### P04H  [portal_10 · portal · heldin_train · HIGH · mechanism: neon-streak-reshape]
- **start_caption** (verbatim): A wide-eyed cartoon character wearing a pointed tinfoil hat, a tactical vest, and crinkled silver boots is in a running pose in a dark, graffiti-covered alleyway illuminated by colorful, glowing vending machines.
- **change_clause**: Streaks of neon light stretch across the alley into long glowing lines that snap and rearrange themselves into the bright outlines of a new scene.
- **end_caption**: A neon-lit cartoon arcade stretches back in rows of glowing game cabinets, their screens flashing pink and blue, a chequered floor reflecting the colours beneath a low starry ceiling.

### P05H  [explosion_1 · explosion · heldout · HIGH · mechanism: dive-through-window]
- **start_caption** (verbatim): A tattooed young man with dark curly hair, wearing a white t-shirt, a red jacket, and light blue jeans, leans back on a red and white motorcycle on an urban street lined with tall brick apartment buildings under overcast daylight.
- **change_clause**: The camera surges forward and dives straight through a lit apartment window across the street, punching cleanly through the glass into the room waiting beyond.
- **end_caption**: A warm living room opens up with a crackling fireplace, two overstuffed sofas facing a low wooden coffee table, framed pictures on the mantel and soft golden lamplight spilling across a patterned rug.

### P06H  [explosion_0 · explosion · heldout · HIGH · mechanism: ocean-wave-crash]
- **start_caption** (verbatim): A long-haired woman wearing glasses, a grey top, a pleated beige mini skirt, black knee-high boots, and a white shoulder bag crosses a city street at a crosswalk illuminated by warm afternoon sunlight between parked cars and overhead power lines.
- **change_clause**: A towering ocean wave sweeps in from the side and crashes across the whole frame, then drags back and pulls the street away with it.
- **end_caption**: A wide sandy beach curves along a turquoise bay at low tide, gentle foam-edged waves rolling in, a lone wooden lifeguard tower standing against a bright sky streaked with thin white clouds.

### P07H  [plasma_explosion_2 · plasma_explosion · heldin_test · HIGH · mechanism: ground-shatter-glass]
- **start_caption** (verbatim): A man with medium-length black hair wearing a black jacket, an olive green t-shirt, and matching cargo pants leans against a bright orange sports car in front of a grey building.
- **change_clause**: The street shatters beneath the car into countless glass shards that lift, spin, and reassemble in mid-air into a different setting.
- **end_caption**: A sleek modern art gallery stretches down a bright white hall, large abstract canvases spaced along the walls, a polished concrete floor reflecting recessed ceiling lights and a single sculptural bench at the centre.

### P08H  [point_cloud_1 · point_cloud · heldout · HIGH · mechanism: sandstorm-sweep]
- **start_caption** (verbatim): A woman with dark hair and white hoop earrings, wearing a white strapless top, leans her arm out the window of a brown classic car parked outdoors against a soft-lit house and a pale green sky in warm evening sunlight.
- **change_clause**: A wall of golden sand sweeps across the frame from the right, blotting out the car entirely, then thins and clears to reveal somewhere new.
- **end_caption**: A vast desert of rippled orange dunes rolls to the horizon under a deep evening sky, long shadows raking across the sand and a lone caravan of camels tracing a distant ridge.

### P09H  [point_cloud_0 · point_cloud · heldout · HIGH · mechanism: train-wipe]
- **start_caption** (verbatim): A person with a white baseball cap, wearing a short-sleeved white shirt and dark pants, sits on a bench in front of a blue and white subway car under bright white light.
- **change_clause**: A train rushes past across the frame in a blur of speeding carriages, and as the last car clears it wipes the platform away behind it.
- **end_caption**: A quiet mountain railway halt sits among tall pines, a small wooden station house with hanging flower baskets beside a single empty track, snow-dusted peaks rising sharply in the crisp morning light.

### P10H  [monstrosity_0 · monstrosity · heldout · HIGH · mechanism: fog-roll-in]
- **start_caption** (verbatim): A young man wearing a dark coat, dark trousers, and a long grey scarf stands on a large rock in a field of white flowers, with snow-capped mountains under an overcast sky in the background.
- **change_clause**: Thick white fog rolls in low across the white flowers and swallows the whole scene, then lifts slowly to uncover a different landscape.
- **end_caption**: A misty green rice terrace steps down a steep valley in wide curving tiers, thin channels of water catching the pale sky, a farmer in a conical hat wading through the flooded paddies.

### P11H  [water_element_3 · water_element · heldin_test · HIGH · mechanism: blizzard-whiteout]
- **start_caption** (verbatim): A blonde woman with long hair, wearing a white earmuff set, a white jacket with fluffy trim, a blue cardigan over a white shirt, and a light blue mini skirt, stands in a snowy mountain landscape under a clear blue sky.
- **change_clause**: A sudden blizzard whips across the slope and whites out the entire frame in swirling snow, then settles and thins to reveal a new place.
- **end_caption**: A warm alpine chalet interior glows with firelight, exposed timber beams overhead, a stone hearth stacked with logs and thick wool blankets draped over a leather couch while steam rises from a mug.

### P12H  [water_element_4 · water_element · heldin_test · HIGH · mechanism: mirror-crack]
- **start_caption** (verbatim): A young man with long black hair, wearing a grey beanie, a bright orange patched jacket, and blue jeans, stands with crossed arms while leaning against a metal pole on a city street under natural daylight.
- **change_clause**: The view cracks like a sheet of mirror glass, the fractured pieces tilting and falling away one by one to expose a different location behind them.
- **end_caption**: A bustling covered market fills with colour, stalls heaped with ripe fruit and hanging lanterns lining a narrow walkway, shoppers weaving past under strings of paper flags in warm afternoon light.

### P13H  [northern_lights_0 · northern_lights · heldout · HIGH · mechanism: aurora-curtain]
- **start_caption** (verbatim): A person with a headlamp and dark goggles, wearing an orange snowsuit and carrying a light gray and orange backpack, stands on a snowy mountain peak at night beneath a dark sky filled with stars.
- **change_clause**: Shimmering curtains of green aurora sweep down and draw across the frame like heavy drapes, then part to either side onto somewhere completely different.
- **end_caption**: A dense tropical jungle steams in humid daylight, broad dripping leaves and hanging vines crowding a narrow trail, a slender waterfall tumbling into a clear pool ringed by mossy grey boulders.

### P14H  [fire_element_2 · fire_element · heldin_test · HIGH · mechanism: flame-wall]
- **start_caption** (verbatim): A woman with long brown hair stands on a black sand beach under a cloudy sky, wearing a long red coat, matching red trousers, and white shoes.
- **change_clause**: A curtain of fire sweeps across the black sand from the left and roars over the frame, then burns down and clears to reveal a new scene.
- **end_caption**: A snow-covered village square rests under soft grey light, timber-framed houses with steep white roofs ringing a frozen fountain, warm lamplight in the windows and a fir tree strung with tiny glowing lights.

### P15H  [fire_element_3 · fire_element · heldin_test · HIGH · mechanism: bird-swarm]
- **start_caption** (verbatim): A shirtless man wearing a black and white varsity jacket, a black cap, and black pants stands motionless in a field of small yellow flowers during an orange and purple sunset.
- **change_clause**: A dense swarm of black birds sweeps up from the field and blots out the whole frame, then scatters apart over a different place.
- **end_caption**: A grand old library reading room stretches upward two storeys, brass-railed balconies of leather-bound books, green banker's lamps glowing along long oak tables and tall arched windows admitting soft afternoon light.

### P16H  [color_rain_2 · color_rain · heldin_test · HIGH · mechanism: ink-flood]
- **start_caption** (verbatim): A blonde woman wearing a black lace top and a cow-print cowboy hat poses in a sunny desert landscape while holding the brim of her hat.
- **change_clause**: A wash of dark ink floods in from every edge of the frame and drowns the desert, then drains rapidly away to leave a new scene behind.
- **end_caption**: A rain-slicked city street glistens at night under neon signs, their pink and blue glow smearing across the wet asphalt, a lone yellow taxi idling at a corner beside a steaming vendor's cart.

### P17H  [color_rain_0 · color_rain · heldin_test · HIGH · mechanism: freeze-and-shatter]
- **start_caption** (verbatim): A woman with long black hair and large sunglasses stands in a warm-lit kitchen, wearing a brown leather jacket, a light blue button-down shirt, and a brown skirt while holding a white telephone receiver to her ear.
- **change_clause**: A sheet of frost races across the kitchen and freezes everything to glittering ice, which then cracks and falls away to reveal a new scene.
- **end_caption**: A quiet snowbound forest clearing rests under pale light, tall firs heavy with snow ringing a frozen pond, a small log cabin with a smoking chimney and soft drifts curving between the trees.

### P18H  [earth_element_6 · earth_element · heldin_test · HIGH · mechanism: earth-splits]
- **start_caption** (verbatim): A young man with a shaved head and a black symbol on his forehead stands in a dusty, rocky canyon under bright daylight, wearing a dark teal robe tied with a rope belt and holding a wooden staff.
- **change_clause**: The canyon floor splits open along a jagged crack and the camera plunges down through the widening chasm into the space below.
- **end_caption**: A vast crystal cavern glows with soft blue light, enormous translucent columns rising from a mirror-still underground lake, glittering veins threading the dark rock walls and a faint mist drifting over the water.

### P19H  [earth_element_4 · earth_element · heldin_test · HIGH · mechanism: puddle-reflection]
- **start_caption** (verbatim): A young man wearing a blue beanie, a blue varsity jacket, baggy white pants, and white sneakers with pink accents stands in front of a white SUV parked on a city street under soft daylight.
- **change_clause**: The camera tips down and pushes into a rain puddle on the street, sinking through its rippling reflection and rising back up somewhere new.
- **end_caption**: A quiet Venetian canal curves between faded ochre buildings, a narrow gondola gliding past shuttered windows and small arched bridges while morning light glints off the calm green water.

### P20H  [train_rush_1 · train_rush · heldout · HIGH · mechanism: light-bloom]
- **start_caption** (verbatim): A man with short dark hair, wearing a black jacket and dark pants, sits hunched forward on a wooden pew inside a dimly lit church with tall arched ceilings and glowing yellow lamps.
- **change_clause**: A swelling bloom of white light rises from the altar and floods outward until it engulfs the frame, then subsides to reveal a different place.
- **end_caption**: A sunlit wildflower meadow rolls gently uphill, waves of red poppies and blue cornflowers nodding in the breeze, a single gnarled oak on the crest and butterflies drifting under a bright open sky.

### P21H  [train_rush_0 · train_rush · heldout · HIGH · mechanism: kaleidoscope-fracture]
- **start_caption** (verbatim): A blonde woman in black sunglasses, wearing a yellow and green long-sleeved jacket and black pants, stands outside on a dark street illuminated by warm streetlights and a camera flash.
- **change_clause**: The frame fractures into a spinning kaleidoscope of repeating wedges that whirl and multiply, then resolve and settle into a single new scene.
- **end_caption**: A calm lakeside dock reaches out over still water at dawn, a small rowboat tied at its end, pine-covered hills mirrored on the glassy surface and thin mist curling above the shallows.

### P22H  [money_rain_3 · money_rain · heldin_test · HIGH · mechanism: wax-slump]
- **start_caption** (verbatim): A person wearing sunglasses, a brown blazer over a patterned shirt, blue trousers, and sandals sits on a green bench inside a boat cabin with wooden floors illuminated by bright sunlight.
- **change_clause**: The whole scene softens and slumps downward as if moulded from warm wax, sliding out of the frame and re-forming smoothly into a different place.
- **end_caption**: A busy fish market spreads under a high steel roof, crates of glistening catch on crushed ice, vendors in rubber aprons calling out and bright overhead lights reflecting off the wet tiled floor.

### P23H  [glitch_1 · glitch · heldout · HIGH · mechanism: cubes-reassemble]
- **start_caption** (verbatim): A dark-haired woman in a sheer grey dress rests her chin on her hands atop a small beige vintage television set that displays a close-up image of her face, sitting on a plain white floor in front of a neutral grey studio backdrop under soft white overhead lighting.
- **change_clause**: The picture breaks apart into thousands of floating cubes that drift and tumble through the air, then click together into a completely different scene.
- **end_caption**: A retro diner glows under warm neon, red vinyl booths along a chrome counter, a chequerboard floor and tall milkshake glasses catching the light while a jukebox glows softly in the corner.

### P24H  [acid_0 · acid · heldout · HIGH · mechanism: smoke-plume]
- **start_caption** (verbatim): A young woman with dark hair, wearing a hooded grey jersey with orange flame graphics and light blue jeans, stands with crossed arms on a deck at night before a illuminated city skyline and bridge over water.
- **change_clause**: A billowing plume of pale smoke rolls across the deck and engulfs the whole frame, then thins and clears over somewhere entirely different.
- **end_caption**: A dense bamboo forest rises in tall green stalks, soft light slanting between them onto a mossy stone path, a small wooden shrine half-hidden among the leaves and gentle mist hanging in the still air.

### P25H  [luminous_gaze_3 · luminous_gaze · heldout · HIGH · mechanism: vines-overgrow]
- **start_caption** (verbatim): A person with dark skin and short curly hair looks over their shoulder on a city street, wearing a grey jacket with a graphic on the back, illuminated by warm sunlight with a yellow construction crane and a modern building in the background.
- **change_clause**: Green vines and broad leaves rush across the frame from the edges and knit together over everything, then peel back to uncover a new place.
- **end_caption**: A grand greenhouse conservatory arches overhead in white iron and glass, lush palms and ferns crowding gravel paths, a tiered stone fountain trickling at the centre under bright diffuse daylight.

### P26H  [saint_glow_3 · saint_glow · heldout · HIGH · mechanism: petal-swarm]
- **start_caption** (verbatim): A young woman with wavy blonde hair crouches on a city sidewalk, resting her chin on her hand, wearing a black sleeveless top, blue jeans, and sandals, with city buildings and traffic in the background.
- **change_clause**: A gust lifts a storm of pink blossom petals across the sidewalk until they fill the frame, then they scatter away to reveal a different setting.
- **end_caption**: A tranquil Japanese garden opens around a still koi pond, an arched red bridge crossing to a mossy islet, carefully raked gravel and a stone lantern set beneath a blooming cherry tree.

### P27H  [live_concert_4 · live_concert · heldout · HIGH · mechanism: rain-sheet]
- **start_caption** (verbatim): A blonde woman wearing dark sunglasses and an oversized black leather jacket with a thick orange stripe down the sleeve is pictured at night on a street with cars in the background.
- **change_clause**: A heavy sheet of rain sweeps down the frame in a shimmering wall of water, then clears from top to bottom onto a completely different place.
- **end_caption**: A cozy mountain cabin porch looks out over a pine valley at dusk, a wooden rocking chair beside a lit lantern, string lights glowing along the eaves and distant peaks receding into blue shadow.

### P28H  [animalization_0 · animalization · heldin_test · HIGH · mechanism: wallpaper-peel]
- **start_caption** (verbatim): A woman in a red bomber jacket poses against a blue studio backdrop, hands on hips.
- **change_clause**: The blue backdrop peels away from one corner like a sheet of wallpaper, rolling steadily across the frame to expose a different scene beneath it.
- **end_caption**: A sunlit Tuscan hillside rolls in golden waves, rows of dark green cypress trees lining a winding dirt road, a terracotta-roofed farmhouse in the distance under a wide, warm afternoon sky.

### P29H  [shadow_3 · shadow · heldin_test · HIGH · mechanism: shadow-swallow]
- **start_caption** (verbatim): A man wearing a dark blue hoodie with "PACE CLUB" printed on the front points his index fingers to his face while baring his decorated teeth against a solid blue background.
- **change_clause**: A deep shadow spreads across the blue background and swallows the frame into darkness, then draws back like a receding tide to reveal a new place.
- **end_caption**: A busy night food street glows with hanging red lanterns, steam rising from rows of open-air stalls, crowds squeezing past bright signage and plastic stools clustered along the narrow lane.

### P30H  [gas_transformation_1 · gas_transformation · heldin_train · HIGH · mechanism: ripple-distortion]
- **start_caption** (verbatim): A bearded man wearing a blue baseball cap, a red sweater, and camouflage shorts crouches on a sunny green field under a clear blue sky with a large tree behind him.
- **change_clause**: The whole frame ripples and warps as if it were the surface of a pond struck by a stone, the wobbling distortion settling into a different scene.
- **end_caption**: A snow-globe alpine ski village nestles in a white valley, chalets with glowing windows lining a snowy street, a chairlift climbing toward pale peaks and skiers gliding past under gently falling snow.

### LOW tier (10)

### P01L  [wireframe_7 · wireframe · heldin_test · LOW]
- **start_caption** (verbatim): A green serum bottle sits among cacti on rocks.
- **change_clause**: The daylight warms gently and a soft highlight blooms on the green glass bottle, while the camera eases slowly closer and a few dust motes drift through the air.
- **end_caption**: The same green serum bottle sits among the cacti on the rocks, now catching a warm soft highlight on its glass as the light turns golden and a few dust motes drift slowly through the air.

### P06L  [explosion_0 · explosion · heldout · LOW]
- **start_caption** (verbatim): A long-haired woman wearing glasses, a grey top, a pleated beige mini skirt, black knee-high boots, and a white shoulder bag crosses a city street at a crosswalk illuminated by warm afternoon sunlight between parked cars and overhead power lines.
- **change_clause**: The warm afternoon light deepens toward a golden hue and the shadows stretch longer across the crosswalk, while a gentle breeze lifts the woman's skirt and hair.
- **end_caption**: The same woman in her grey top and beige skirt crosses the crosswalk between the parked cars, the afternoon light now a deeper gold with long soft shadows reaching across the street.

### P09L  [point_cloud_0 · point_cloud · heldout · LOW]
- **start_caption** (verbatim): A person with a white baseball cap, wearing a short-sleeved white shirt and dark pants, sits on a bench in front of a blue and white subway car under bright white light.
- **change_clause**: The bright platform light softens to a warmer tone and the camera pushes in slightly, while the reflections on the subway car windows shift and glimmer gently.
- **end_caption**: The same person in the white cap sits on the bench in front of the blue and white subway car, the harsh light now warmer and softer while the camera frames them a little more closely.

### P10L  [monstrosity_0 · monstrosity · heldout · LOW]
- **start_caption** (verbatim): A young man wearing a dark coat, dark trousers, and a long grey scarf stands on a large rock in a field of white flowers, with snow-capped mountains under an overcast sky in the background.
- **change_clause**: The overcast light warms gently toward gold and a soft mist begins to drift low across the white flowers, while the camera eases almost imperceptibly closer.
- **end_caption**: The same young man in his long grey scarf stands on the rock among the white flowers, the snow-capped mountains behind him now touched with warm golden light as thin mist drifts low across the field.

### P11L  [water_element_3 · water_element · heldin_test · LOW]
- **start_caption** (verbatim): A blonde woman with long hair, wearing a white earmuff set, a white jacket with fluffy trim, a blue cardigan over a white shirt, and a light blue mini skirt, stands in a snowy mountain landscape under a clear blue sky.
- **change_clause**: The clear blue sky warms toward a pale afternoon gold and light snowflakes begin to drift down, while a gentle breeze stirs the fluffy trim of the woman's jacket.
- **end_caption**: The same blonde woman in her white fur-trimmed jacket stands in the snowy mountain landscape, the sky now a warm pale gold as fine snowflakes drift down and a light breeze lifts her hair.

### P14L  [fire_element_2 · fire_element · heldin_test · LOW]
- **start_caption** (verbatim): A woman with long brown hair stands on a black sand beach under a cloudy sky, wearing a long red coat, matching red trousers, and white shoes.
- **change_clause**: The overcast sky brightens softly and a warm rim of sunlight edges the clouds, while a light wind lifts the hem of the red coat and ruffles the woman's hair.
- **end_caption**: The same woman in her long red coat stands on the black sand beach, the cloudy sky now brightening with a warm glow at its edges as a gentle sea breeze stirs her hair and coat.

### P17L  [color_rain_0 · color_rain · heldin_test · LOW]
- **start_caption** (verbatim): A woman with long black hair and large sunglasses stands in a warm-lit kitchen, wearing a brown leather jacket, a light blue button-down shirt, and a brown skirt while holding a white telephone receiver to her ear.
- **change_clause**: The warm kitchen light brightens gently and a soft push-in eases the camera a little closer, while steam begins to curl upward from a mug on the counter.
- **end_caption**: The same woman stands in her warm-lit kitchen holding the white telephone receiver, framed slightly closer now, with gentle steam rising from a mug on the counter as the afternoon light grows warmer.

### P19L  [earth_element_4 · earth_element · heldin_test · LOW]
- **start_caption** (verbatim): A young man wearing a blue beanie, a blue varsity jacket, baggy white pants, and white sneakers with pink accents stands in front of a white SUV parked on a city street under soft daylight.
- **change_clause**: The daylight shifts toward a warm late-afternoon glow and the shadows lengthen slightly, while the camera pushes in gently and a few leaves drift down past the parked SUV.
- **end_caption**: The same young man in his blue varsity jacket stands in front of the white SUV on the city street, now lit by warm late-afternoon sun with long soft shadows and a few leaves drifting down.

### P23L  [glitch_1 · glitch · heldout · LOW]
- **start_caption** (verbatim): A dark-haired woman in a sheer grey dress rests her chin on her hands atop a small beige vintage television set that displays a close-up image of her face, sitting on a plain white floor in front of a neutral grey studio backdrop under soft white overhead lighting.
- **change_clause**: The soft studio light dims slightly to a cooler blue and a faint warm glow rises from the television screen, while the camera drifts a little closer.
- **end_caption**: The same dark-haired woman rests her chin on her hands atop the vintage television, the studio now lit in a cooler blue with a soft warm glow spilling from the screen showing her face.

### P28L  [animalization_0 · animalization · heldin_test · LOW]
- **start_caption** (verbatim): A woman in a red bomber jacket poses against a blue studio backdrop, hands on hips.
- **change_clause**: The studio lights slowly warm from cool white to a soft golden tone and the blue backdrop deepens a shade, while a gentle breeze stirs the woman's hair.
- **end_caption**: The same woman in her red bomber jacket stands against the blue studio backdrop, now bathed in warm golden light that softens the shadows on her face while her hair lifts gently in a light breeze.

## Rejected eligible endpoints and why

- **acid_1** (acid, heldout): dropped for redundancy — its setting duplicates an already-selected scene type (most eligible clips are single-person portraits); kept the more visually distinct example. Caption: "A woman with brown hair with bangs, wearing orange sunglasses, a white strap top, gold earrings, and a gold ch…"
- **animalization_2** (animalization, heldin_test): dropped for redundancy — its setting duplicates an already-selected scene type (most eligible clips are single-person portraits); kept the more visually distinct example. Caption: "A young woman with long, dark hair crouches in the middle of a sunny asphalt street lined with parked cars and…"
- **animalization_3** (animalization, heldin_train): dropped for redundancy — its setting duplicates an already-selected scene type (most eligible clips are single-person portraits); kept the more visually distinct example. Caption: "A young man in a bucket hat and red t-shirt stands in front of a corner coffee house."
- **color_rain_3** (color_rain, heldin_train): class 'color_rain' already at the 2-clip cap with more distinctive scenes. Caption: "A young man with short hair, wearing orange-tinted sunglasses and a striped brown and beige long-sleeve shirt,…"
- **cotton_cloud_0** (cotton_cloud, heldout): dropped for redundancy — its setting duplicates an already-selected scene type (most eligible clips are single-person portraits); kept the more visually distinct example. Caption: "A young blonde woman wearing sunglasses, a black leather jacket with yellow and white stripes, a white crop to…"
- **earth_element_1** (earth_element, heldin_train): class 'earth_element' already at the 2-clip cap with more distinctive scenes. Caption: "A man with dreadlocks, wearing a light blue denim jacket with "SUPREME N.Y.C." printed on the front over a whi…"
- **gas_transformation_6** (gas_transformation, heldin_test): dropped for redundancy — its setting duplicates an already-selected scene type (most eligible clips are single-person portraits); kept the more visually distinct example. Caption: "A bearded man wearing a white cap, a green cable-knit quarter-zip sweater, blue pants, and white sneakers sits…"
- **gas_transformation_7** (gas_transformation, heldin_test): dropped for redundancy — its setting duplicates an already-selected scene type (most eligible clips are single-person portraits); kept the more visually distinct example. Caption: "A man in a cowboy hat, red jacket and cream trousers leans against a brown corrugated wall."
- **glitch_0** (glitch, heldout): dropped for redundancy — its setting duplicates an already-selected scene type (most eligible clips are single-person portraits); kept the more visually distinct example. Caption: "A person with long curly black hair, wearing black sunglasses, a red track jacket, a beige shirt, and white pa…"
- **illustration_scene_1** (illustration_scene, heldin_train): dropped for redundancy — its setting duplicates an already-selected scene type (most eligible clips are single-person portraits); kept the more visually distinct example. Caption: "A young man with a mustache, wearing a white bandana, a white t-shirt, grey shorts, and red and black sneakers…"
- **illustration_scene_4** (illustration_scene, heldin_test): dropped for redundancy — its setting duplicates an already-selected scene type (most eligible clips are single-person portraits); kept the more visually distinct example. Caption: "A blonde woman with sunglasses and a headscarf blows a kiss with both hands at her cheeks."
- **illustration_scene_7** (illustration_scene, heldin_test): dropped for redundancy — its setting duplicates an already-selected scene type (most eligible clips are single-person portraits); kept the more visually distinct example. Caption: "A young man in a leather jacket and denim shorts leans against a railing by a stone wall."
- **live_concert_0** (live_concert, heldout): dropped for redundancy — its setting duplicates an already-selected scene type (most eligible clips are single-person portraits); kept the more visually distinct example. Caption: "A smiling man and woman wearing matching black graphic t-shirts and bright pink sunglasses pose between beige …"
- **money_rain_1** (money_rain, heldin_test): dropped for redundancy — its setting duplicates an already-selected scene type (most eligible clips are single-person portraits); kept the more visually distinct example. Caption: "A young woman with long, wavy blonde hair sits on a city street curb, resting her chin on her hand while weari…"
- **money_rain_2** (money_rain, heldin_train): dropped for redundancy — its setting duplicates an already-selected scene type (most eligible clips are single-person portraits); kept the more visually distinct example. Caption: "A man in sunglasses and a dark green work shirt stands in a wood-paneled room, hands in pockets."
- **mystification_0** (mystification, heldin_test): dropped for redundancy — its setting duplicates an already-selected scene type (most eligible clips are single-person portraits); kept the more visually distinct example. Caption: "A young Black man with short blonde hair, glasses, and a pearl necklace sits on a chair in front of a draped b…"
- **mystification_1** (mystification, heldin_test): dropped for redundancy — its setting duplicates an already-selected scene type (most eligible clips are single-person portraits); kept the more visually distinct example. Caption: "A man with blonde hair and glasses, wearing a red and black sleeveless shirt, brown pants, and black boots, si…"
- **northern_lights_1** (northern_lights, heldout): dropped for redundancy — its setting duplicates an already-selected scene type (most eligible clips are single-person portraits); kept the more visually distinct example. Caption: "A dark-haired man in a black jacket and brown trousers sits in a dark armchair on a wooden deck, looking out a…"
- **plasma_explosion_3** (plasma_explosion, heldin_test): dropped for redundancy — its setting duplicates an already-selected scene type (most eligible clips are single-person portraits); kept the more visually distinct example. Caption: "A young man with short blonde hair, wearing a red, blue, and yellow jacket over a white t-shirt, black pants, …"
- **polygon_1** (polygon, heldin_train): dropped for redundancy — its setting duplicates an already-selected scene type (most eligible clips are single-person portraits); kept the more visually distinct example. Caption: "A man wearing a tan baseball cap, sunglasses, a pink patterned headscarf, a brown t-shirt over a white long-sl…"
- **polygon_4** (polygon, heldin_test): dropped for redundancy — its setting duplicates an already-selected scene type (most eligible clips are single-person portraits); kept the more visually distinct example. Caption: "A young woman with her hair in a bun stands on a cobblestone street in bright daylight, wearing an oversized w…"
- **polygon_8** (polygon, heldin_test): dropped for redundancy — its setting duplicates an already-selected scene type (most eligible clips are single-person portraits); kept the more visually distinct example. Caption: "A young woman with long black hair, wearing bright pink undereye makeup, silver chain necklaces, and a blue ja…"
- **portal_4** (portal, heldin_test): class 'portal' already at the 2-clip cap with more distinctive scenes. Caption: "A cartoon character with teal hair and a green apron rests their head on their hand while holding a pen over a…"
- **shadow_1** (shadow, heldin_train): dropped for redundancy — its setting duplicates an already-selected scene type (most eligible clips are single-person portraits); kept the more visually distinct example. Caption: "A young woman with short, wavy dark hair stands in a narrow alleyway between adobe buildings under warm golden…"
- **shadow_10** (shadow, heldin_test): dropped for redundancy — its setting duplicates an already-selected scene type (most eligible clips are single-person portraits); kept the more visually distinct example. Caption: "A smiling woman in a leather jacket, white top and light jeans sits against an iron fence."
- **super_fast_run_0** (super_fast_run, heldin_test): dropped for redundancy — its setting duplicates an already-selected scene type (most eligible clips are single-person portraits); kept the more visually distinct example. Caption: "A young woman with dark hair stands in a narrow paved alleyway, wearing a black Adidas soccer jersey tucked in…"
- **super_fast_run_10** (super_fast_run, heldin_train): dropped for redundancy — its setting duplicates an already-selected scene type (most eligible clips are single-person portraits); kept the more visually distinct example. Caption: "A man in sunglasses, a dark green hoodie and white cap stands in a bright industrial kitchen."
- **super_fast_run_3** (super_fast_run, heldin_test): dropped for redundancy — its setting duplicates an already-selected scene type (most eligible clips are single-person portraits); kept the more visually distinct example. Caption: "A sweaty young Black man with short curly hair and a serious expression looks directly at the camera while wea…"
- **wireframe_0** (wireframe, heldin_test): class 'wireframe' already at the 2-clip cap with more distinctive scenes. Caption: "The lower legs of a person wearing grey trousers and black Salomon sneakers are suspended in the air in an ind…"
- **x_ray_0** (x_ray, heldout): dropped for redundancy — its setting duplicates an already-selected scene type (most eligible clips are single-person portraits); kept the more visually distinct example. Caption: "A woman with shoulder-length dark hair, wearing light gray over-ear headphones and a black jacket, stands in a…"
- **x_ray_1** (x_ray, heldout): dropped for redundancy — its setting duplicates an already-selected scene type (most eligible clips are single-person portraits); kept the more visually distinct example. Caption: "A young woman with long wavy blonde hair, wearing a dark blue strapless dress with a small bow detail, sits ca…"

_Skipped by construction (not eligible): all `humanvid_*` (no clip on disk), all `davis` sources, all two-sided (scene-changing) classes, and `sided==one` endpoints whose source was outside {heldin_test, heldout, heldin_train}._


> Owner edit 2026-09-08 (fable check): P02H start clip `wireframe_2` (t-shirt with crude printed text) replaced by `super_fast_run_10` (bright industrial kitchen, interior start scene); paper-fold mechanism kept.
