from utils.logging_utils import Experiment
from utils.results_utils import (
    generate_main_results,
    generate_main_results_less_samples,
    generate_main_results_ood_table,
    load_df_folder,
    seed_all,
)

seed_all(0)

LOG_FOLDER = None
LOG_FOLDER = ""

if LOG_FOLDER is not None:
    df = load_df_folder(LOG_FOLDER)
    if df.iloc[0]["experiment"] == Experiment.MAIN_TABLE.name:
        _, table = generate_main_results(df)
        print("")
        print(table)
    elif df.iloc[0]["experiment"] == Experiment.LESS_SAMPLES.name:
        _, table = generate_main_results_less_samples(df)
        print("")
        print(table)
    elif df.iloc[0]["experiment"] == Experiment.OOD_INSIGHT.name:
        _, table = generate_main_results_ood_table(df)
        print("")
        print(table)
    elif (
        df.iloc[0]["experiment"] == Experiment.D3_ABLATION_NO_CRITIC.name
        or df.iloc[0]["experiment"] == Experiment.D3_ABLATION_NO_MEMORY.name
    ):
        _, table = generate_main_results(df)
        print("")
        print(table)
else:
    pass
