# Ananya v2 — Seed Matrix

Target dataset: 25-35 images. All diversity gates must pass before training.

## Trigger Word
`AnyV2X9` — non-semantic, no real-name pull. Reserved for v2.

## Identity (LOCKED — never caption)
Face shape, eye color/shape, nose, lips, skin tone, ethnicity, body type, age (always 23).

## Variables (caption these)
Outfit, hair, expression, pose, scene, lighting direction, focal length, depth-of-field, aesthetic mode, makeup, jewelry, eye contact.

---

## Diversity Gate Targets

### Mode distribution (~30 total)
| Mode | Count | % |
|---|---|---|
| Extreme close-up of eyes | 2 | 7% |
| Closeup (face-shoulder) | 14 | 47% |
| Medium (waist-up) | 9 | 30% |
| Full body | 5 | 17% |

### Resolution distribution
| Aspect | Resolution | Count | % |
|---|---|---|---|
| Square | 1024×1024 | 18 | 60% |
| Vertical (9:16) | 832×1216 | 9 | 30% |
| Horizontal (3:2) | 1216×832 | 3 | 10% |

### Outfit distribution (8 distinct, ≤15% each)
| Outfit | Count | Category |
|---|---|---|
| Silk saree (emerald with gold border) | 3 | Ethnic |
| Lehenga (mirror work + dupatta) | 2 | Ethnic |
| Cotton kurti + palazzo | 2 | Ethnic |
| Black bodycon midi dress | 4 | Western |
| High-waist jeans + crop top | 4 | Western |
| Tailored blazer over satin cami | 3 | Western |
| Casual sundress (floral) | 3 | Western |
| Plain white tee + denim shorts | 3 | Neutral |
| Simple wrap top + skirt | 3 | Neutral |
| White anarkali + dupatta | 2 | Ethnic |
| **Total** | **30** | |

### Hair distribution (3 styles + entropy)
| Style | Count | % |
|---|---|---|
| Loose waves side-parted | 15 | 50% |
| Half-up | 9 | 30% |
| Low bun | 6 | 20% |

**Entropy variants** (≥1 each):
- Wind movement / mid-flip
- Backlit hair (rim catching strands)
- Wet/textured hair
- Messy strands / flyaways
- Tight bun vs loose bun

### Pose class distribution (variance gate)
| Pose class | Min count | Examples |
|---|---|---|
| sitting | 4 | chair, ledge, café seat, floor-leaning |
| walking_motion | 4 | mid-stride, hair-flip, looking-back-over-shoulder |
| interacting | 4 | holding coffee cup, phone, handbag, leaning on table, sunglasses gesture |
| standing_static | 6 (capped) | facing camera, weight on one leg, hands at side |
| candid_distracted | 3 | adjusting clothes, looking away, lost in thought |
| Total | 21+ | (rest distributed) |

### Camera angles (≥3 each)
- Front facing
- Profile L
- Profile R
- Three-quarter L
- Three-quarter R
- Slight high elevation
- Slight low elevation

### Focal length distribution
| Focal | Count | % | Use case |
|---|---|---|---|
| 24mm environmental | 3 | 10% | Wide street, full scene |
| 35mm lifestyle | 11 | 35% | OOTD, candid |
| 50mm portrait | 11 | 35% | Standard portrait |
| 85mm fashion editorial | 5 | 20% | Tight portrait, editorial |

### Depth-of-field distribution (defeats v1 always-bokeh)
| DOF | Count | % |
|---|---|---|
| Deep focus / sharp background (f/8+) | 15 | 50% |
| Medium DOF | 9 | 30% |
| Shallow bokeh (only editorial close-up) | 6 | 20% |

### Lighting direction (≥2 each)
- Front-lit (flat soft)
- Side-lit (key from left or right)
- Top-lit (overhead, café pendant)
- Rim-lit (backlight rim)
- Backlit (golden hour silhouette)
- Mixed practical (warm tungsten + window daylight)

### Palette diversity (≥1 each)
- Cool tones (overcast, blue hour)
- Dark scenes (low-light)
- Monochrome / desaturated
- Colorful / saturated
- High saturation pop
- Muted daylight

### Camera sensor realism (≥3 distinct)
- DSLR editorial (sharp, premium grade)
- iPhone candid (slight compression, mobile color)
- Low-light mobile (visible grain, noise)
- Compressed social-media realism (Instagram-look)

### Crop diversity (≥4 imperfect crops)
- Imperfect crops (subject slightly off)
- Partial body truncation (hand or leg cut)
- Off-center framing (rule of thirds)
- Edge framing (subject pushed to edge)

### Motion blur realism (≥3 with subtle blur)
- Mild handheld softness
- Subtle movement blur (walking, hair)
- Low-light grain

### Background semantic diversity
- ≥3 luxury (hotel lobby, premium café)
- ≥4 ordinary (street, plain wall, regular café)
- ≥3 cluttered (real apartments, lived-in spaces)
- ≥3 minimal (studio, neutral wall)

### Aesthetic modes (≥1 each, no luxury monoculture)
- Editorial fashion (premium magazine look)
- Candid phone-camera realism
- Soft luxury lifestyle
- High-contrast rooftop (golden hour)
- Indoor tungsten warm
- Muted cloudy daylight

### Makeup distribution
| Mode | Count | Description |
|---|---|---|
| No makeup | 4 | Natural skin |
| Natural daytime | 12 | Subtle blush, lip tint |
| Editorial glam | 5 | Defined eyes, lips |
| Matte daylight | 5 | Modern matte finish |
| Glossy evening | 4 | Glossy lips, highlight |

### Jewelry disentanglement
- ≥3 with NO jewelry
- ≥3 with earrings only
- ≥3 with necklace only
- ≥3 with bangles
- ≥3 with mixed (multiple)
- Vary earring style (studs, hoops, drops)

### Eye contact diversity
- ≥8 direct gaze (camera)
- ≥6 off-camera gaze (looking elsewhere)
- ≥4 distracted gaze (engaged with environment)
- ≥3 downward gaze (looking down/at hands)

### Hand visibility
- 40-50% of dataset has visible hands
- ≥2 close-up shots showing hands clearly
- Varied gestures (holding, gesturing, resting, hair-touching)

### Asymmetry + imperfection (anti-AI-tell)
- ≥3 uneven smile
- ≥2 slight squint
- ≥3 uneven posture / weight shift
- ≥2 tired/sleepy expression
- ≥2 controlled imperfection (uneven lighting, slight fatigue)

### Occlusion robustness
- ≥1 sunglasses occluding eyes
- ≥2 hair partially across face
- ≥2 hand near or partially blocking face
- ≥1 object partially blocking (cup, phone)

### Emotional expression entropy
- ≥3 candid smile
- ≥2 awkward laugh
- ≥2 neutral / flat
- ≥2 annoyed neutral
- ≥2 sleepy / tired
- ≥3 soft contemplative

### Non-fashion anchor (grounding)
- ≥2 in kitchen / cooking
- ≥1 on couch (casual indoor)
- ≥1 in ordinary domestic setting

### Same-outfit multi-condition cluster (carousel consistency)
For 3 outfits, generate 3 conditions each (lighting/pose/location varies, outfit identical):
- Cluster A: black bodycon midi × {hotel lobby warm, rooftop golden hour, café candid}
- Cluster B: silk saree emerald × {hotel lobby, rooftop dusk, garden}
- Cluster C: jeans + crop top × {street day, café indoor, apartment}

These 9 cluster images count toward dataset total.

---

## Source Distribution (per ChatGPT #1)

| Source | Count | Method |
|---|---|---|
| Stock anchor + faceswap + IC-Light | 17 (57%) | Unsplash/Pexels face-obscured → ReActor/FaceFusion + IC-Light |
| v1 LoRA generation | 7 (23%) | bootstrap_seeds_v2.py with diverse prompts |
| Flux Kontext / Qwen-Image-Edit | 6 (20%) | Clothing swap on strong v1 closeups → ethnic wear |
| **Total** | **30** | |

---

## 30-Row Seed Matrix

Each row = one image with all attributes pre-specified. Generate/curate to fit.

| ID | Mode | Resolution | Outfit | Hair | Hair Entropy | Pose Class | Camera Angle | Focal | DOF | Lighting | Palette | Sensor | Crop | Aesthetic | Makeup | Jewelry | Eye Contact | Asymmetry | Occlusion | Emotion | Source | Hands |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 01 | Closeup | 1024² | Black bodycon | Loose waves | Mid-flip | standing_static | three-quarter R | 85mm | shallow | warm tungsten side | warm | DSLR | clean | editorial fashion | editorial glam | drop earrings | direct | uneven smile | none | candid smile | stock_swap | no |
| 02 | Closeup | 832×1216 | Silk saree emerald | Loose waves | Backlit rim | sitting | front | 50mm | medium | golden hour back | warm saturated | DSLR | clean | luxury lifestyle | natural daytime | gold studs+necklace | direct | none | none | soft contemplative | edit_inpaint | yes (saree pleat) |
| 03 | Eyes XCU | 1024² | n/a | Loose waves | none | n/a | front | 85mm | shallow | front-lit soft | neutral | DSLR | tight | editorial | natural | drop earring visible | direct | slight squint | hair across | flat | stock_swap | no |
| 04 | Medium | 832×1216 | Cotton kurti + palazzo | Half-up | none | walking_motion | three-quarter L | 35mm | deep f/8 | front-lit daylight | colorful | iPhone candid | imperfect | candid phone realism | natural | no jewelry | off-camera | uneven posture | hair across | candid smile | stock_swap | yes (gesturing) |
| 05 | Full body | 832×1216 | Lehenga | Low bun | tight | standing_static | front | 35mm | deep f/8 | mixed practical | warm | DSLR | clean | editorial | editorial glam | full set | direct | none | none | soft smile | edit_inpaint | yes (dupatta hold) |
| 06 | Closeup | 1024² | Plain white tee | Loose waves | messy flyaways | candid_distracted | profile L | 50mm | medium | side-lit window | muted | iPhone candid | off-center | candid phone | no makeup | no jewelry | distracted | uneven smile | none | tired sleepy | stock_swap | no |
| 07 | Medium | 1024² | Tailored blazer + cami | Loose waves | none | interacting | three-quarter R | 50mm | medium | top-lit pendant | dark | DSLR | clean | soft luxury | matte daylight | thin necklace | direct | none | hand near face | confident | stock_swap | yes (hand near collar) |
| 08 | Closeup | 1024² | High-waist jeans + crop top | Half-up | none | candid_distracted | three-quarter L | 35mm | deep f/8 | front-lit daylight | colorful | iPhone candid | imperfect | candid phone | natural | hoops | off-camera | none | none | candid laugh | stock_swap | yes (hair touching) |
| 09 | Full body | 832×1216 | High-waist jeans + crop top | Loose waves | wind | walking_motion | three-quarter R | 24mm | deep f/8 | side-lit street | colorful saturated | iPhone candid | edge framing | candid phone | natural | hoops | off-camera | uneven posture | none | candid smile | stock_swap | yes (one in pocket) |
| 10 | Closeup | 1024² | Black bodycon | Low bun | loose tendrils | standing_static | front | 85mm | shallow | front-lit soft | dark | DSLR | clean | editorial | glossy evening | studs | direct | none | none | confident | v1_gen | no |
| 11 | Medium | 832×1216 | Casual sundress | Loose waves | wind | walking_motion | profile R | 35mm | deep f/8 | backlit golden hour | warm | DSLR | clean | high-contrast rooftop | natural | no jewelry | off-camera | uneven posture | hair across | candid smile | stock_swap | yes (skirt holding) |
| 12 | Closeup | 1024² | Silk saree | Low bun | tight | sitting | three-quarter L | 50mm | medium | warm tungsten side | warm | DSLR | clean | indoor tungsten | editorial glam | full set | direct | none | none | soft smile | edit_inpaint | no |
| 13 | Medium | 1024² | Simple wrap top + skirt | Loose waves | none | interacting | three-quarter R | 50mm | medium | top-lit café | warm | DSLR | clean | soft luxury | natural | thin necklace | distracted | none | hand on cup | candid | stock_swap | yes (holding cup) |
| 14 | Full body | 832×1216 | Plain white tee + denim shorts | Half-up | none | sitting | front | 35mm | deep f/8 | front-lit daylight | muted | iPhone candid | off-center | candid phone | no makeup | bangles | off-camera | uneven posture | none | flat | stock_swap | yes (resting on knees) |
| 15 | Closeup | 1216×832 | Black bodycon | Loose waves | none | candid_distracted | profile R | 85mm | shallow | rim-lit | dark | DSLR | clean | editorial | matte daylight | drops | downward | none | none | contemplative | v1_gen | no |
| 16 | Closeup | 1024² | Tailored blazer + cami | Half-up | none | interacting | three-quarter L | 50mm | medium | front-lit window | neutral | DSLR | clean | soft luxury | natural | studs | direct | uneven smile | hand near face | candid smile | stock_swap | yes (adjusting collar) |
| 17 | Medium | 832×1216 | Casual sundress | Loose waves | none | sitting | front | 50mm | medium | side-lit garden | colorful | DSLR | clean | luxury lifestyle | natural | bangles | direct | none | none | soft smile | stock_swap | yes (in lap) |
| 18 | Closeup | 1024² | Cotton kurti | Half-up | messy | candid_distracted | three-quarter R | 50mm | medium | mixed practical | warm | iPhone candid | imperfect | candid phone | no makeup | no jewelry | off-camera | tired sleepy | hair across | sleepy | edit_inpaint | no |
| 19 | Full body | 832×1216 | White anarkali + dupatta | Low bun | none | standing_static | front | 35mm | deep f/8 | top-lit | warm saturated | DSLR | clean | editorial | editorial glam | full set | direct | none | none | confident | edit_inpaint | yes (dupatta) |
| 20 | Closeup | 1024² | Plain white tee | Loose waves | wet | sitting | front | 85mm | shallow | side-lit window | muted | DSLR | clean | candid phone | no makeup | no jewelry | distracted | none | hair across | flat | stock_swap | no |
| 21 | Medium | 1216×832 | High-waist jeans + crop top | Loose waves | none | walking_motion | three-quarter R | 35mm | deep f/8 | front-lit street | colorful | low-light mobile | edge framing | candid phone | natural | hoops | direct | uneven posture | none | candid laugh | stock_swap | yes (mid-stride) |
| 22 | Eyes XCU | 1024² | n/a | n/a | n/a | n/a | front | 85mm | shallow | front-lit | neutral | DSLR | tight | editorial | editorial glam | drop earring | direct | slight squint | none | flat | stock_swap | no |
| 23 | Closeup | 1024² | Lehenga (close-up) | Half-up | none | sitting | three-quarter L | 85mm | shallow | warm tungsten | warm | DSLR | clean | indoor tungsten | editorial glam | full set | direct | none | none | confident | edit_inpaint | no |
| 24 | Medium | 832×1216 | Black bodycon (cluster A1) | Loose waves | none | standing_static | three-quarter R | 50mm | medium | warm tungsten lobby | warm | DSLR | clean | soft luxury | matte daylight | studs | direct | uneven posture | none | confident | v1_gen | no |
| 25 | Medium | 832×1216 | Black bodycon (cluster A2) | Loose waves | none | walking_motion | front | 50mm | medium | golden hour rooftop | warm saturated | DSLR | clean | high-contrast rooftop | matte daylight | studs | off-camera | uneven posture | hair across | candid smile | v1_gen | no |
| 26 | Medium | 832×1216 | Black bodycon (cluster A3) | Loose waves | none | candid_distracted | profile L | 35mm | medium | side-lit café | warm | iPhone candid | off-center | candid phone | matte daylight | studs | distracted | none | none | flat | v1_gen | yes (on cup) |
| 27 | Closeup | 1024² | Silk saree (cluster B1) | Low bun | tight | sitting | front | 50mm | medium | warm tungsten lobby | warm | DSLR | clean | indoor tungsten | editorial glam | full set | direct | none | none | soft smile | edit_inpaint | no |
| 28 | Closeup | 1024² | Simple wrap top | Loose waves | none | candid_distracted | profile R | 50mm | medium | side-lit kitchen | warm | iPhone candid | off-center | candid phone (NON-FASHION) | no makeup | no jewelry | distracted | tired sleepy | none | sleepy | stock_swap | yes (cooking) |
| 29 | Medium | 1024² | Plain white tee | Half-up | none | sitting | front | 35mm | deep f/8 | front-lit window | muted | iPhone candid | off-center | candid phone (NON-FASHION couch) | no makeup | bangles | downward | uneven posture | none | flat | stock_swap | yes (on phone) |
| 30 | Full body | 832×1216 | Casual sundress | Loose waves | wind backlit | walking_motion | three-quarter L | 24mm | deep f/8 | backlit golden hour | warm | DSLR | edge framing | high-contrast rooftop | natural | bangles | off-camera | uneven posture | hair across | candid smile | stock_swap | yes (skirt) |

---

## Validation Checklist (before training)

- [ ] Total count: 25-35 (target 30)
- [ ] Mode mix: ~50% closeup, ~30% medium, ~20% full body, 2-3 eye XCU
- [ ] Resolution mix: 60% 1024², 30% vertical, 10% horizontal
- [ ] Outfit: 8+ distinct, none >15%, ethnic + Western + neutral covered
- [ ] Hair: 3 styles + ≥1 entropy variant per category
- [ ] Pose class: ≥3 each of sitting/walking/interacting, ≤6 standing static
- [ ] Camera angles: ≥3 each (front, profile L+R, ¾ L+R)
- [ ] Focal: 24/35/50/85mm distribution roughly 10/35/35/20%
- [ ] DOF: 50% deep / 30% medium / 20% shallow
- [ ] Lighting direction: ≥2 each of front/side/top/rim/back/mixed
- [ ] Palette: ≥1 each of cool/dark/mono/colorful/saturated/muted
- [ ] Sensor: ≥3 distinct (DSLR/iPhone/low-light)
- [ ] Crop: ≥4 imperfect/off-center/edge
- [ ] Motion blur: ≥3 with subtle softness/grain
- [ ] Background semantic: ≥3 luxury, ≥4 ordinary, ≥3 cluttered, ≥3 minimal
- [ ] Aesthetic: ≥1 each of 6 modes
- [ ] Makeup: ≥4 no-makeup, distribution across 5 modes
- [ ] Jewelry: ≥3 no jewelry, varied per row
- [ ] Eye contact: ≥8 direct, ≥6 off, ≥4 distracted, ≥3 down
- [ ] Hands: 40-50% visible
- [ ] Asymmetry/imperfection: ≥10 rows with at least one imperfection
- [ ] Occlusion: ≥6 rows with some occlusion
- [ ] Expression entropy: distribution across smile/laugh/neutral/tired/contemplative
- [ ] Non-fashion anchor: ≥3 grounding shots (kitchen/couch/domestic)
- [ ] Same-outfit cluster: 3 outfits × 3 conditions = 9 cluster images
- [ ] Source mix: 50-60% stock_swap, 20-30% v1_gen, 20% edit_inpaint
- [ ] CLIP cosine similarity audit: avg < 0.85, no pair > 0.92
- [ ] Texture integrity: no waxy, no over-smooth
