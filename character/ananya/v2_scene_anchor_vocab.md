# Ananya v2 — Scene Anchor Vocabulary

Caption vocabulary for geographic, lighting, focal, and DOF descriptions. Use exact phrases — consistent vocabulary prevents semantic drift across captions.

---

## Geographic Anchors

### Bengaluru / South India
| Scene | Caption phrase |
|---|---|
| Indiranagar street | `narrow leafy street, terracotta-tiled rooftops, warm street lamp glow, Bengaluru evening` |
| Koramangala café | `modern concrete café interior, exposed brick, tropical monsoon greenery visible through window, warm tungsten pendant lights` |
| Rooftop terrace | `open rooftop terrace, distant Bengaluru city skyline, hazy golden hour pink sky, humid evening air` |
| UB City adjacent | `polished marble plaza, luxury retail façade, warm indirect uplighting, urban South India evening` |
| Brigade Road sidewalk | `wide pedestrian street, mix of old and new buildings, warm shopfront glow, casual urban India` |
| Indoor apartment | `modern mid-range Bengaluru apartment, white walls, warm LED panel light, urban domestic interior` |

### Pan-India Luxury / Generic Premium
| Scene | Caption phrase |
|---|---|
| Hotel lobby | `luxury hotel lobby, Italian marble floors, warm tungsten chandelier light, high ceiling, South Asian premium hospitality` |
| Hotel corridor | `hotel corridor, warm wall sconces, patterned carpet, soft hallway light` |
| Hotel room vanity | `hotel room vanity mirror, warm bulb ring lights, white marble countertop, luxury hotel bathroom adjacent` |
| Rooftop bar | `rooftop bar terrace, city skyline at dusk, candlelit table, ambient warm string lights` |
| Airport terminal | `modern Indian airport terminal, large glass windows, bright overhead lighting, departure lounge` |

### Outdoor / Street India
| Scene | Caption phrase |
|---|---|
| Heritage district street | `old-city narrow lane, weathered pastel walls, cobblestones, soft diffused cloudy daylight, North India heritage aesthetic` |
| Metro station | `modern Indian metro station, clean grey concrete, fluorescent overhead, urban commuter crowd blurred background` |
| Flower market | `Indian flower market, marigold garlands in background, warm ambient daylight, outdoor ground-level shot` |
| Beach promenade | `Goa or Mumbai seaside promenade, sea wall railing, golden late afternoon sun, open horizon sky` |
| Garden / park | `Lalbagh botanical garden, lush tropical greenery, dappled sunlight through canopy, soft overcast garden light` |

### Studio / Neutral
| Scene | Caption phrase |
|---|---|
| Studio neutral | `softbox studio, neutral grey seamless paper backdrop, even diffused light, no background detail` |
| Studio white | `bright white studio, high-key lighting, white seamless background, clean editorial` |
| Studio coloured | `studio, warm terracotta seamless backdrop, soft side-key light` |

---

## Lighting Anchors

### Direction
| Direction | Caption phrase |
|---|---|
| Front flat | `flat front lighting, even illumination, no strong shadows` |
| Side key | `side key light from camera left, soft shadow on far side of face` |
| Rembrandt | `Rembrandt lighting, triangular cheek highlight, dramatic shadow` |
| Top/overhead | `overhead ambient light, slight top light, soft undereye shadow` |
| Rim / backlit | `strong rim light from behind, hair illuminated, background blown slightly` |
| Backlit golden hour | `backlit golden hour, warm haze, sun directly behind, hair catch light` |
| Mixed ambient | `mixed ambient light, window from right, soft fill from left, no hard shadows` |

### Quality
| Quality | Caption phrase |
|---|---|
| Soft diffused | `soft diffused light, no hard shadows, overcast window light` |
| Hard specular | `hard directional light, defined shadows, harsh high-contrast` |
| Golden hour | `golden hour warm light, amber tones, long shadows` |
| Tungsten warm | `warm tungsten indoor light, orange-amber cast, indoor evening` |
| Neon / mixed | `neon accent light, purple-pink ambient, dark moody interior` |
| Fluorescent harsh | `harsh fluorescent ceiling light, flat cool tone, realistic candid indoor` |
| Low-light mobile | `low-light mobile photography, visible noise grain, candid underlit indoor` |
| Daylight overcast | `flat daylight, overcast sky, soft even light, cool slightly desaturated` |
| Dusk blue hour | `blue hour dusk, deep blue sky, warm artificial lights beginning, twilight` |

---

## Focal Length Anchors

| Focal | Caption phrase | Use |
|---|---|---|
| 24mm environmental | `wide environmental framing, 24mm equivalent, slight perspective distortion, full scene visible` | 10% — group/context shots |
| 35mm lifestyle | `35mm lifestyle framing, natural perspective, subject prominent in scene, candid distance` | 35% — medium and full body |
| 50mm portrait | `50mm natural portrait framing, minimal compression, clean subject-background separation` | 35% — waist-up, face |
| 85mm editorial | `compressed telephoto portrait, 85mm equivalent, background compression, editorial bokeh` | 20% — close-up, editorial |

---

## Depth of Field Anchors

| DOF | Caption phrase | Dataset target |
|---|---|---|
| Deep focus / sharp bg | `sharp background, environmental detail visible, f/8 deep focus, iPhone 15 Pro Max realism, candid phone camera shot` | 50% |
| Medium DOF | `moderate background blur, some background detail visible, f/2.8 portrait, DSLR natural` | 30% |
| Shallow bokeh | `shallow depth of field, strong background bokeh, f/1.4 editorial, subject sharp, background melted` | 20% |

---

## Sensor / Camera Realism Anchors

| Mode | Caption phrase |
|---|---|
| DSLR premium | `DSLR fashion photography, Sony A7IV quality, clean sensor, professional grade` |
| iPhone candid | `iPhone 15 Pro Max shot, candid, phone camera realism, natural skin tone rendering` |
| Low-light mobile | `handheld low-light mobile shot, visible grain, slight motion softness, candid` |
| Film grain | `subtle film grain, analog texture, slightly desaturated, editorial film aesthetic` |

---

## Aesthetic Mode Anchors

| Aesthetic | Caption phrase |
|---|---|
| Editorial fashion | `editorial fashion photography, strong composition, deliberate pose, magazine aesthetic` |
| Candid phone realism | `candid lifestyle photography, natural unstaged moment, phone camera realism, authentic` |
| Soft luxury lifestyle | `soft luxury lifestyle, warm tones, aspirational but approachable, Instagram fashion` |
| High-contrast rooftop | `high-contrast rooftop portrait, dramatic shadows, bold golden light, strong silhouette` |
| Indoor tungsten warm | `indoor warm lifestyle, tungsten light, cozy ambient, domestic or hotel intimate` |
| Muted cloudy daylight | `muted cloudy daylight, flat soft light, desaturated palette, indie editorial` |
| UGC style | `user-generated content style, imperfect framing, natural light, authentic everyday India` |
| Dark moody | `dark moody interior, low-key lighting, shadows prominent, dramatic atmosphere` |

---

## Crop / Framing Anchors

| Crop | Caption phrase |
|---|---|
| Perfect centered | *(no tag needed — default)* |
| Off-center rule-of-thirds | `subject off-center, rule of thirds framing, breathing room on one side` |
| Partial truncation | `slightly cropped at frame edge, natural candid framing, not perfectly centered` |
| Imperfect handheld | `imperfect handheld framing, slight dutch angle, authentic candid composition` |
| Edge of frame | `subject near frame edge, wide negative space, environmental emphasis` |

---

## DOF + Sensor Combinations (pre-built for common use)

| Combo label | Full caption phrase |
|---|---|
| Street candid | `sharp background, environmental detail visible, f/8 deep focus, iPhone 15 Pro Max realism, candid phone camera shot, 35mm lifestyle framing` |
| Studio editorial | `shallow depth of field, strong background bokeh, f/1.4 editorial, subject sharp, 85mm equivalent, DSLR fashion photography` |
| Hotel lifestyle | `moderate background blur, some background detail visible, f/2.8 portrait, warm tungsten indoor light, 50mm natural portrait framing` |
| Rooftop golden hour | `backlit golden hour, warm haze, shallow depth of field, 85mm compressed telephoto portrait, editorial bokeh, hair catch light` |
| Domestic candid | `sharp background, environmental detail visible, f/8 deep focus, iPhone 15 Pro Max realism, indoor warm LED, 35mm lifestyle framing` |
| Low-light mobile | `low-light mobile photography, visible noise grain, candid underlit indoor, handheld motion softness, 35mm equivalent` |

---

## Caption Template

```
<shot type>, <focal length phrase> of AnyV2X9 seen from <camera angle> at <elevation>, 
with <hair style and state>. She is <pose/action> and expressing <emotion>. 
<Lighting phrase>, <DOF phrase>, <aesthetic mode phrase>, <geographic anchor phrase>.
```

### Shot types
- `Extreme close-up of AnyV2X9's face`
- `Close-up portrait`
- `Waist-up shot`
- `Three-quarter length shot`
- `Full body shot`
- `Wide environmental shot`

### Camera angles
- `front facing` / `three-quarter view from camera left` / `three-quarter view from camera right`
- `profile from camera left` / `profile from camera right`
- `slight high angle, looking slightly down at subject`
- `slight low angle, looking slightly up at subject`

### Elevation
- `eye level` / `slightly elevated` / `slightly low` / `ground level`

---

## Token Frequency Anti-patterns

**Avoid over-clustering these tokens** — if they appear in >40% of captions they bind semantically:

| Token | Risk if overused | Mitigation |
|---|---|---|
| `warm tungsten` | binds to "premium/luxury" only | vary: overcast/fluorescent/golden hour |
| `luxury hotel lobby` | scene variety collapses | cap at 2-3 images max |
| `bokeh` / `shallow depth` | AI tell baked in | 50% deep focus, explicit f/8 |
| `editorial fashion` | aesthetic locked | vary UGC / candid / soft luxury |
| `loose waves` | hair bakes | 50/30/20 distribution, explicit hair states |
| `golden hour` | lighting variety collapses | distribute: tungsten/overcast/fluorescent/blue hour |
| `Mumbai` / `Delhi` | geography binds | use Bengaluru anchors + generic where needed |
| `sharp detailed face` | prompt dependency bakes | omit after v2 — LoRA should handle |
| `AnyV2X9` | must be FIRST token always | verify every caption — keep_tokens=1 |

---

## Vocabulary Consistency Rules

1. Always `AnyV2X9` (capital V, numeral 2, X, numeral 9) — never abbreviate or vary
2. Focal length always as `Xmm equivalent` — not `Xmm lens` or `Xmm focal`
3. DOF phrase always before aesthetic phrase in template
4. Geographic anchor always last in description block
5. Hair state (wind/wet/messy/flat/backlit) always immediately after hair style
6. Emotion: `soft smile` / `neutral expression` / `direct gaze, serious` / `candid laugh, eyes crinkled` / `distracted, looking away` / `tired, flat expression` — never just `happy` or `sad`
7. Outfit omits color-light conflation: `emerald saree` not `yellow-green saree under warm light` — light goes in lighting phrase
