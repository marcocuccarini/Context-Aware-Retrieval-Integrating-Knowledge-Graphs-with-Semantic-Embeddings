
from Classes.ranking_document import *
from Classes.run_expermiment import *


# 🚀 Run all experiments with custom output directory
if __name__ == "__main__":
    output_folder = "Results/Rankings/"
    cities = ["florence", "rome", "venice"]
    output_folder_name = ["SmallBERT","BERT","SmallSPADE","SPADE","BM25"]

    for city in cities:


        run_dense_cosine_experiment(city, model_name="all-MiniLM-L6-v2", output_dir=output_folder+output_folder_name[0])
        run_dense_cosine_experiment(city, model_name="all-mpnet-base-v2", output_dir=output_folder+output_folder_name[1])
        run_spade_experiment(city, model_name="sentence-transformers/all-MiniLM-L6-v2", output_dir=output_folder+output_folder_name[2])
        run_spade_experiment(city, model_name="sentence-transformers/all-mpnet-base-v2", output_dir=output_folder+output_folder_name[3])
        run_bm25_experiment(city, output_dir=output_folder+output_folder_name[4])


    for directory in output_folder_name:
        for city in cities:
            run_city_experiment(city, directory)
