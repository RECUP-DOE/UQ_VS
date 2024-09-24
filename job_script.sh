# Meta flags
SECONDS=0
trial_array=(0 ) #1 2 3 4)
num_models_array=(10 )#20 30)
AS_dim_array=(5 ) #10 15 20)
prop_array=('DRD2' ) #'GSK3B')

echo "training property predictor model"

for prop_name in "${prop_array[@]}"; do
    python train_surrogate.py --prop_name=$prop_name
done

echo "finished training property predictor model"

for prop_name in "${prop_array[@]}"; do
    
    for AS_dim in "${AS_dim_array[@]}"; do
        echo "Property =${prop_name}, AS dimnesion=${AS_dim}"
        python run_active_subspace_construction.py --prop_name=$prop_name --AS_dim=$AS_dim
        python run_vi_training.py --prop_name=$prop_name --AS_dim=$AS_dim

        for num_models in "${num_models_array[@]}"; do
            for trial in "${trial_array[@]}"; do
                python run_screening_AS_pred.py --prop_name=$prop_name --AS_dim=$AS_dim --trial=$trial --num_models=$num_models

            done
        done
    done
done



echo "Elapsed Time: $SECONDS seconds"