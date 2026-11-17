# caveat nrun configs/actvae/conditional.yaml --n 5 -ni
caveat nrun configs/actvae/vae.yaml --n 5 -ni
caveat nrun configs/actvae/cpvae.yaml --n 5 -ni
# caveat nrun configs/actvae/are.yaml --n 5 -ni

# ablations
caveat nrun configs/actvae/ablation/cvae.yaml --n 5 -ni
caveat nrun configs/actvae/ablation/cpvae-ni.yaml --n 5 -ni
caveat nrun configs/actvae/ablation/cpvaeL2.yaml --n 5 -ni
caveat nrun configs/actvae/ablation/cpvaeL6.yaml --n 5 -ni
caveat nrun configs/actvae/ablation/cpvaeL8.yaml --n 5 -ni
caveat nrun configs/actvae/ablation/cpvaeL10.yaml --n 5 -ni


caveat nrun configs/actvae/scenarios/veh_extrapolation.yaml --n 5 -ni


# caveat batch configs/actvae/scenarios/cvae_sampled_scenario.yaml
# caveat batch configs/actvae/scenarios/cvae_labels_scenario.yaml
# caveat batch configs/actvae/scenarios/cvae_transfer_scenario.yaml

