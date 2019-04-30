#!/usr/bin/env bash

# Predict on test data
python3 predict_test.py \
--epochs 125 \
--pretrain_weights /workdir/bes/weights/unet_resnext_50_stage_2_1/fold_0.hdf5 \
--fold 0,1,2,3,4 \
--learning_rate 0.00005 \
--input_size 224 \
--resize_size 192 \
--batch_size 28 \
--loss_function lovasz \
--callback reduce_lr \
--early_stop_patience 15 \
--reduce_lr_factor 0.5 \
--reduce_lr_patience 8 \
--reduce_lr_min 0.0000125 \
--network unet_resnext_50 \
--alias _lovasz_stage_2_2 \
--prediction_weights fold_{}.hdf5 \
--prediction_folder fold_{}

python3 predict_test.py \
--epochs 20 \
--n_snapshots 1 \
--pretrain_weights /workdir/bes/weights/unet_resnext_50_lovasz_stage_2_2/fold_{}.hdf5 \
--fold 0,1,2,3,4 \
--learning_rate 0.00005 \
--input_size 224 \
--resize_size 192 \
--batch_size 28 \
--loss_function lovasz \
--callback snapshot \
--network unet_resnext_50 \
--alias _lovasz_stage_2_3 \
--prediction_weights fold_{}.hdf5 \
--prediction_folder fold_{}

python3 predict_test.py \
--epochs 20 \
--n_snapshots 1 \
--pretrain_weights /workdir/bes/weights/unet_resnext_50_lovasz_stage_2_3/fold_{}.hdf5 \
--fold 0,1,2,3,4 \
--learning_rate 0.00005 \
--input_size 224 \
--resize_size 192 \
--batch_size 28 \
--loss_function lovasz \
--callback snapshot \
--network unet_resnext_50 \
--alias _lovasz_stage_2_4 \
--prediction_weights fold_{}.hdf5 \
--prediction_folder fold_{}

python3 predict_test.py \
--epochs 25 \
--n_snapshots 1 \
--pretrain_weights /workdir/bes/weights/unet_resnext_50_lovasz_stage_2_4/fold_{}.hdf5 \
--fold 0,1,2,3,4 \
--learning_rate 0.00005 \
--input_size 224 \
--resize_size 192 \
--batch_size 28 \
--loss_function lovasz \
--callback snapshot \
--network unet_resnext_50 \
--alias _lovasz_stage_2_5 \
--prediction_weights fold_{}.hdf5 \
--prediction_folder fold_{}
