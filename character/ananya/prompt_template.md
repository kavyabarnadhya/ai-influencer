# Ananya v2 Carousel Prompt Template

**Target aesthetic** (from `seeds_v2/training_canonical/` reference):
candid editorial fashion photography — 50mm vertical, natural lighting, real urban locations,
authentic skin texture, casual confident poses. "Shot by a friend with a good camera" — not
studio-perfect, not over-airbrushed.

---

## Slot order (FLUX weighs first tokens most)

```
1. PHOTOGRAPHIC ANCHOR     — pins aesthetic, must be first
2. FRAMING + FOCAL         — close-up / waist-up / full-body, 50mm / 35mm
3. SUBJECT + HAIR          — "23-year-old South Asian woman with <hair>"
4. OUTFIT (specific)       — fabric, color, cut, details
5. JEWELRY / ACCESSORIES   — specific items
6. POSE + EXPRESSION       — concrete photographer language
7. SCENE (named place)     — Mumbai rooftop, Lalbagh garden, etc.
8. LIGHTING                — direction + color temperature
9. DOF + LENS              — f/1.8 shallow / f/2.8 moderate / f/8 deep
10. SKIN + GRAIN ANCHOR    — defeats AI plastic look
```

## Template

```
photorealistic [aesthetic_anchor],
[framing] of a 23-year-old South Asian woman with [hair_style + state],
wearing [specific outfit with fabric + color + cut],
[jewelry list],
[pose + expression],
[named scene] visible behind her with [scene depth detail],
[lighting direction + color + intensity],
[DOF f/value], sharp focus on face,
natural skin texture with visible pores, light film grain, no AI plastic smoothing
```

## Aesthetic anchor phrases (slot 1)

- `photorealistic Instagram editorial fashion photograph, shot on Sony A7IV`
- `photorealistic candid lifestyle photograph, shot on Fujifilm X-T5`
- `photorealistic candid iPhone photo by a friend, slightly imperfect framing`
- `photorealistic editorial fashion magazine shot, Vogue India style`
- `photorealistic golden hour street photography, 50mm prime`

## Vocabulary library

### Framing
- `tight close-up portrait`
- `waist-up portrait, slight tilt`
- `full body three-quarter pose`
- `full body from slight low angle`
- `over-the-shoulder back view`

### Hair states
- `dark hair in loose beachy waves over right shoulder`
- `dark hair in low messy bun with face-framing strands`
- `dark hair straight middle-parted falling past shoulders`
- `dark hair in high ponytail with wispy baby hairs`
- `dark hair pulled back in sleek low chignon`

### Outfits — ethnic
- `deep maroon silk Banarasi saree with intricate gold zari border, draped pleated, fitted black blouse`
- `sage green chanderi cotton saree with thin gold border, white cotton blouse`
- `pink georgette lehenga choli with mirror-work embroidery, matching dupatta draped over left shoulder`
- `mustard yellow chikankari kurta with white palazzo pants, dupatta over arm`
- `black raw silk Anarkali with gold thread embroidery on bodice`

### Outfits — western
- `fitted high-waisted blue denim jeans and tucked-in cream linen shirt`
- `rust-red floral midi sundress with thin spaghetti straps, open back`
- `oversized cream cashmere sweater and pleated tartan mini skirt`
- `black satin slip midi dress with cowl neckline`
- `vintage Levi's 501 cropped jeans and white tank top under unbuttoned linen shirt`

### Outfits — fusion / premium
- `deep V-neck black satin slip dress, plunging neckline, fitted bias cut`
- `bralette top with high-waisted printed silk wrap skirt, indo-western fusion`
- `sheer black saree with crystal embellished blouse, modern drape`
- `velvet co-ord set with cropped bustier top and high-waisted flared pants`

### Jewelry
- `gold jhumka earrings`, `delicate gold pendant necklace`
- `kundan choker with matching earrings`
- `silver oxidized hoop earrings with stack of thin bangles`
- `single statement gold cuff bracelet`
- `dark cat-eye sunglasses`, `tortoiseshell aviator sunglasses`

### Pose + expression
- `weight shifted on right leg, hands resting at sides, soft direct gaze at camera`
- `mid-laugh head tilted back, eyes closed, candid moment`
- `looking over right shoulder with playful smirk, hand brushing hair`
- `walking forward toward camera, slight motion blur in trailing hand`
- `seated leaning forward elbow on knee, contemplative serious expression`
- `hand on hip, other hand adjusting hair, confident editorial stance`

### Scenes — named urban
- `Mumbai high-rise rooftop at golden hour, hazy city skyline visible behind`
- `narrow old Delhi heritage lane with weathered pastel walls and cobblestones`
- `Bengaluru Indiranagar cafe terrace with string lights and brick walls`
- `Lalbagh botanical garden pathway at dusk with banyan trees`
- `wide pedestrian street in Connaught Place, mix of colonial and modern buildings`
- `Pondicherry French quarter sidewalk with bougainvillea-covered yellow walls`
- `Jaipur havelis courtyard with sandstone arches and mirror inlay`
- `luxury hotel marble lobby with crystal chandeliers and Italian flooring`

### Lighting
- `warm golden hour backlight from right rim-lighting her hair, soft fill bouncing off pavement`
- `overcast diffused daylight, even soft tones, slight cool cast`
- `harsh midday sun creating dappled shadows through trees`
- `warm tungsten interior light from above, amber wash`
- `mixed window light from camera left, soft fill from white wall right`
- `blue hour ambient with neon shopfront accent lighting in background`

### DOF
- `f/1.8 shallow depth of field, soft melted background bokeh`
- `f/2.8 moderate background blur, subject sharp, scene readable`
- `f/4 mid-range, foreground sharp with some background detail`
- `f/8 deep focus, full scene in focus, environmental portrait`

### Skin + grain anchors (always append)
- `natural skin texture with visible pores and subtle imperfections`
- `light 35mm film grain`
- `no plastic AI smoothing, no airbrushed look, authentic skin`

---

## Worked example

```
photorealistic Instagram editorial fashion photograph shot on Sony A7IV 50mm,
waist-up portrait of a 23-year-old South Asian woman with dark hair in loose
beachy waves over right shoulder,
wearing deep maroon silk Banarasi saree with intricate gold zari border draped
pleated and fitted black blouse,
gold jhumka earrings and delicate gold pendant,
slight serene smile looking directly at camera, hand gently holding saree pleats,
Mumbai high-rise rooftop at golden hour with hazy city skyline visible behind her,
warm orange backlight from upper right rim-lighting her hair, soft warm fill on face,
f/1.8 shallow depth of field with soft melted background bokeh, sharp focus on face,
natural skin texture with visible pores, light 35mm film grain,
no plastic AI smoothing, no airbrushed look
```

---

## Tips

- **One long line per slide** in the prompts file — line breaks above are for readability only.
- **Repeat "South Asian" and "dark hair"** in every slide — FLUX schnell drifts toward European defaults if missing.
- **Cleavage / premium prompts**: use editorial fashion descriptors (`deep V-neck`, `plunging neckline`, `bias cut`). Avoid clinical or euphemistic language — FLUX trained on editorial fashion photography handles fashion terms naturally.
- **Body consistency**: stays from anchor img2img seed. Don't describe body type ("slim", "curvy") in slide prompts — let anchor handle it.
- **If slide fails**: first try lowering denoise 0.05; if still failing, rewrite scene with more concrete/named elements.
