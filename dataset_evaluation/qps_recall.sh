#/bin/bash

# datasets=("bigann-1M" "deep-1M" "msturing-1M" "ssnpp-1M" "openai-1M" "text2image-1M" "msmarco-1M" "wikipedia-1M")
# large_datasets=("bigann-10M" "bigann-10M" "wikipedia-10M" "ssnpp-10M" "ssnpp-100M")
large_datasets=("bigann-100M")
datasets=("text2image-1M" "msmarcowebsearch-1M" "wikipedia-1M")
# datasets=("bigann-1M")
# for dataset in "${datasets[@]}";
# do 
#     python3 run_qps_recall_experiment.py -p -datasets [$dataset] -groups ['"Beam Search", "Greedy Search with Early Stopping", "Doubling Search with Early Stopping", "FAISS"'] -graph_name $dataset
# done  



# for dataset in "${large_datasets[@]}";
# do 
    # python3 run_qps_recall_experiment.py -datasets [$dataset] -groups ['"All, Beam Search, Average Precision"','"All, Doubling Search with Early Stopping, Average Precision"','"All, Greedy Search with Early Stopping, Average Precision"'] -graph_name $dataset"_ap" -p -g
# done

for dataset in "${datasets[@]}";
do 
    python3 run_qps_recall_earlystopping_experiment.py -datasets [$dataset] -groups ['"Doubling Search"','"Doubling Search with Early Stopping"'] -graph_name $dataset"_doubling_earlystopping" -p 
    python3 run_qps_recall_earlystopping_experiment.py -datasets [$dataset] -groups ['"Greedy Search"','"Greedy Search with Early Stopping"'] -graph_name $dataset"_greedy_earlystopping" -p 
done


# for dataset in "${large_datasets[@]}";
# do 
    # python3 run_qps_recall_earlystopping_experiment.py -datasets [$dataset] -groups ['"All, Doubling Search, Average Precision"','"All, Doubling Search with Early Stopping, Average Precision"'] -graph_name $dataset"_doubling_ap" -p -g
    # python3 run_qps_recall_earlystopping_experiment.py -datasets [$dataset] -groups ['"All, Doubling Search, Pointwise Precision"','"All, Doubling Search with Early Stopping, Pointwise Precision"'] -graph_name $dataset"_doubling_pp" -p -g
    # python3 run_qps_recall_earlystopping_experiment.py -datasets [$dataset] -groups ['"All, Greedy Search, Pointwise Precision"','"All, Greedy Search with Early Stopping, Pointwise Precision"'] -graph_name $dataset"_greedy_pp" -p -g
    # python3 run_qps_recall_earlystopping_experiment.py -datasets [$dataset] -groups ['"All, Greedy Search, Average Precision"','"All, Greedy Search with Early Stopping, Average Precision"'] -graph_name $dataset"_greedy_ap" -p -g
# done