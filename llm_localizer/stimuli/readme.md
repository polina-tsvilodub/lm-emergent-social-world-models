Directory for stimuli for localization of various networks.

* `language`: the original language network localization data from AlKhamissi et al (2025)
* `tom`: the original ToM network localization data from AlKhamissi et al (2025)

To attempt to improve upon the contrasts in the original materials, various synthetitc stimuli for ToM localization were generated. The stimuli are in the original ToM localization format (as done in the localization paper). Stories and questions are in two different files. The correct answer is not provided (hence only usable for localization, not for evaluation).
The synthetic data is all in `synthetic_stimuli`, and was created based on few-shot prompts with examples from original human studies. The original human materials are in `original_stimuli`.

Each of the localizers described in the paper consist of the following synthetic stimuli:

- `deceptive_dec_synthetic`:
- `deceptive_iro_synthetic`:
- `deceptive_lit_synthetic`:
- `deceptive_mls_synthetic`:
- `exp1_ha_synthetic`:
- `exp1_mi_synthetic`:
- `exp2_d_synthetic_d_2a`:
- `exp2_fb_synthetic_fb_2a`:
- `exp2_fp_synthetic_fp_2a`:
- `exp2_h_synthetic_2a`:
- `exp2_nh_synthetic_2a`:
- `mechanical_inference_synthetic`:
- `moral_intent_control_synthetic`:
- `moral_intent_tom_synthetic`:
- `strategic_games_eo_synthetic_paired`:
- `strategic_games_synthetic_eb`:
- `strategic_games_synthetic_eo`: