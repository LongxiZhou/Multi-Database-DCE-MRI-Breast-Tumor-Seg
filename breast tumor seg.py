import prediction.predict_rescaled as predictor
import numpy as np
import visualization.visiualize_2d.visualize_breast_tumor as visualize

rescaled_array = np.load('/DCE-MRI_data/1724535_unknown.npz')['array']
input_array = rescaled_array[:, :, :, 0: 3]
gt_array = rescaled_array[:, :, :, 3]
predict_array = predictor.predict_breast_tumor_dcm_mri(input_array, check_point_top_dict='/trained_models/')
visualize.analysis_prediction(input_array, predict_array, gt_array)



