#!/usr/bin/env python3

import cv2
import numpy as np
import depthai as dai
from depthai import CameraBoardSocket
from depthai import Clock
from depthai import ColorCameraProperties
from depthai import DeviceInfo
from depthai import IMUSensor
from depthai import ImgDetections
from depthai import ImgFrame
from depthai import MonoCameraProperties
from depthai import Pipeline
from depthai import RawStereoDepthConfig
from depthai import StereoDepthConfig
from depthai import StereoDepthProperties
from depthai import TrackedFeatures
from depthai import VideoEncoderProperties
from depthai.node import StereoDepth
import time
import os
from pathlib import Path

print("enter a descriptive name of the camera")
name = input()

def getDisparityFrame(disp, cvColorMap):
    
    disp = (disp/255).astype(np.uint8)
    disp = cv2.applyColorMap(disp, cvColorMap)

    return disp

RIGHT_STREAM_NAME = "right"
DEPTH_RIGHT_STREAM_NAME = "depth"
FEATURES_STREAM_NAME = "features"

frame_rate = 10


home = Path.home()
save_folder = f"{home}/OddBotVisionLocal/output/camera_test/{int(time.time())}/"
print(save_folder)
save_count = 0

class test:
  def __init__(self):
    pass
    
parameters = test()
parameters.depth_confidence_threshold = 220
parameters.depth_disparity_shift = 0
parameters.depth_unit_multiplier = 5000
parameters.depth_min_brightness = 16
parameters.depth_max_brightness = 240
parameters.depth_hole_filling_radius = 2
parameters.depth_speckle_threshold = 2
parameters.depth_speckle_range = 25
parameters.depth_min_range = 0.1
parameters.depth_max_range = 100.0

pipeline = Pipeline()

#color = pipeline.createColorCamera()
#color.setBoardSocket(CameraBoardSocket.RGB)
#color.setFps(frame_rate)
#color.setResolution(ColorCameraProperties.SensorResolution.THE_4_K)
#color.setVideoSize(parameters.color_video_width, parameters.color_video_height)
#color.setInterleaved(False)
#color.initialControl.setSharpness(parameters.sharpness)
#color.initialControl.setLumaDenoise(parameters.luma_denoise)
#color.initialControl.setChromaDenoise(parameters.chroma_denoise)
#color.initialControl.setAutoFocusLensRange(parameters.focus_range_infinity_position, parameters.focus_range_macro_position)
#color.initialControl.setAutoExposureLimit(parameters.exposure_limit_us)

#color_downsample_script = pipeline.createScript()
#color_downsample_script.setScript(create_downsample_script(DOWNSAMPLE_SCRIPT_STREAM_IN, DOWNSAMPLE_SCRIPT_STREAM_OUT, 2))

#color_resize = pipeline.createImageManip()
#color_resize.initialConfig.setResize(parameters.color_preview_width, parameters.color_preview_height)
#color_resize.initialConfig.setFrameType(ImgFrame.Type.BGR888p)

#hi_res_encoder = pipeline.createVideoEncoder()
#hi_res_encoder.setProfile(VideoEncoderProperties.Profile.MJPEG)
#hi_res_encoder.setQuality(parameters.high_resolution_image_quality)

left = pipeline.createMonoCamera()
left.setBoardSocket(CameraBoardSocket.CAM_B)
left.setFps(frame_rate)
left.setResolution(MonoCameraProperties.SensorResolution.THE_400_P)  # lowest resolution for more short range accuracy
left.initialControl.setAutoExposureRegion(startX = 120, startY= 100, width=400, height=300) 

right = pipeline.createMonoCamera()
right.setBoardSocket(CameraBoardSocket.CAM_C)
right.setFps(frame_rate)
right.setResolution(MonoCameraProperties.SensorResolution.THE_400_P)  # lowest resolution for more short range accuracy
right.initialControl.setAutoExposureRegion(startX = 120, startY= 100, width=400, height=300) 

depth = pipeline.createStereoDepth()
#depth.setDepthAlign(CameraBoardSocket.CAM_C)
depth.setDefaultProfilePreset(StereoDepth.PresetMode.HIGH_ACCURACY)
#depth.setExtendedDisparity(True)  # more accuracy for short range
depth.setLeftRightCheck(True)  # required for depth alignment
depth.enableDistortionCorrection(True)
depth.initialConfig.setConfidenceThreshold(parameters.depth_confidence_threshold)
depth.initialConfig.setMedianFilter(StereoDepthProperties.MedianFilter.MEDIAN_OFF)  # recommended for high accuracy
depth.initialConfig.setSubpixel(True)
depth.initialConfig.setSubpixelFractionalBits(5)
depth.initialConfig.setBilateralFilterSigma(0)
depth.initialConfig.setDisparityShift(parameters.depth_disparity_shift)
depth.initialConfig.setDepthUnit(RawStereoDepthConfig.AlgorithmControl.DepthUnit.CUSTOM)
depth_config = depth.initialConfig.get()
depth_config.algorithmControl.customDepthUnitMultiplier = parameters.depth_unit_multiplier
depth_config.postProcessing.brightnessFilter.minBrightness = parameters.depth_min_brightness
depth_config.postProcessing.brightnessFilter.maxBrightness = parameters.depth_max_brightness
depth_config.postProcessing.decimationFilter.decimationFactor = 1  # disables decimation filter
depth_config.postProcessing.decimationFilter.decimationMode = StereoDepthConfig.PostProcessing.DecimationFilter.DecimationMode.PIXEL_SKIPPING  # default
depth_config.postProcessing.spatialFilter.enable = True
depth_config.postProcessing.spatialFilter.alpha = 0.5  # default
depth_config.postProcessing.spatialFilter.delta = 0  # default
depth_config.postProcessing.spatialFilter.holeFillingRadius = parameters.depth_hole_filling_radius
depth_config.postProcessing.spatialFilter.numIterations = 1  # default
depth_config.postProcessing.speckleFilter.enable = True
depth_config.postProcessing.speckleFilter.differenceThreshold = parameters.depth_speckle_threshold
depth_config.postProcessing.speckleFilter.speckleRange = parameters.depth_speckle_range
depth_config.postProcessing.temporalFilter.enable = False
depth_config.postProcessing.temporalFilter.alpha = 0.4  # default
depth_config.postProcessing.temporalFilter.delta = 0  # default
depth_config.postProcessing.temporalFilter.persistencyMode = StereoDepthConfig.PostProcessing.TemporalFilter.PersistencyMode.VALID_2_IN_LAST_4  # default
depth_config.postProcessing.thresholdFilter.minRange = int(parameters.depth_min_range * parameters.depth_unit_multiplier)
depth_config.postProcessing.thresholdFilter.maxRange = int(parameters.depth_max_range * parameters.depth_unit_multiplier)
depth_config.postProcessing.filteringOrder = [RawStereoDepthConfig.PostProcessing.Filter.MEDIAN,
                                          RawStereoDepthConfig.PostProcessing.Filter.DECIMATION,
                                          RawStereoDepthConfig.PostProcessing.Filter.SPECKLE,
                                          RawStereoDepthConfig.PostProcessing.Filter.SPATIAL,
                                          RawStereoDepthConfig.PostProcessing.Filter.TEMPORAL, ]  # default
depth.initialConfig.set(depth_config)

features = pipeline.createFeatureTracker()

features.setHardwareResources(numShaves=2, numMemorySlices=2)

#model_name = parameters.detection_model_names[parameters.detection_model_name_idx]
#model_config_path = MODEL_FOLDER / (model_name + '.json')
#with open(model_config_path) as fp:
#model_config = json.load(fp)
#model_blob_path = MODEL_FOLDER / (model_name + '.blob')
#labels = model_config['labels']
#coordinate_size = model_config['coordinate_size']
#anchors = model_config['anchors']
#anchor_masks = model_config['anchor_masks']
#iou_threshold = model_config['iou_threshold']
#detector_confidence_threshold = model_config['confidence_threshold']

#detection = pipeline.createYoloDetectionNetwork()
#detection.setBlobPath(model_blob_path)
#detection.setAnchors(anchors)
#detection.setAnchorMasks(anchor_masks)
#detection.setConfidenceThreshold(detector_confidence_threshold)
#detection.setNumClasses(len(labels))
#detection.setCoordinateSize(coordinate_size)
#detection.setIouThreshold(iou_threshold)
#detection.setNumInferenceThreads(2)
#detection.input.setBlocking(False)
#detection.input.setQueueSize(1)

#imu = pipeline.createIMU()
#imu.enableIMUSensor(IMUSensor.GAME_ROTATION_VECTOR, 50)
# imu.enableIMUSensor(IMUSensor.LINEAR_ACCELERATION, 50) # Temporary disabled for raw logging
#imu.enableIMUSensor(IMUSensor.ACCELEROMETER_RAW, 50)  # Temporary enabled for raw logging
#imu.enableIMUSensor(IMUSensor.GYROSCOPE_RAW, 50)  # Temporary enabled for raw logging
#imu.enableIMUSensor(IMUSensor.MAGNETOMETER_RAW, 50)  # Temporary enabled for raw logging
#imu.setBatchReportThreshold(1)
#imu.setMaxBatchReports(10)

#control_in = pipeline.createXLinkIn()
#color_out = pipeline.createXLinkOut()
#hi_res_out = pipeline.createXLinkOut()
right_out = pipeline.createXLinkOut()
depth_out = pipeline.createXLinkOut()
features_out = pipeline.createXLinkOut()
#detection_out = pipeline.createXLinkOut()
#imu_out = pipeline.createXLinkOut()

#control_in.setStreamName(CONTROL_STREAM_NAME)
#color_out.setStreamName(COLOR_STREAM_NAME)
#hi_res_out.setStreamName(HI_RES_STREAM_NAME)
right_out.setStreamName(RIGHT_STREAM_NAME)
depth_out.setStreamName(DEPTH_RIGHT_STREAM_NAME)
features_out.setStreamName(FEATURES_STREAM_NAME)
#detection_out.setStreamName(DETECTIONS_STREAM_NAME)
#imu_out.setStreamName(IMU_STREAM_NAME)

depth_out.input.setBlocking(False)
depth_out.input.setQueueSize(1)

#control_in.out.link(color.inputControl)
#color.video.link(color_downsample_script.inputs[DOWNSAMPLE_SCRIPT_STREAM_IN])
#color_downsample_script.outputs[DOWNSAMPLE_SCRIPT_STREAM_OUT].link(color_resize.inputImage)
#color_downsample_script.outputs[DOWNSAMPLE_SCRIPT_STREAM_OUT].link(hi_res_encoder.input)
#color_resize.out.link(color_out.input)
#color_resize.out.link(detection.input)
#hi_res_encoder.bitstream.link(hi_res_out.input)
left.out.link(depth.left)
right.out.link(depth.right)
depth.depth.link(depth_out.input)
depth.rectifiedRight.link(right_out.input)
depth.rectifiedRight.link(features.inputImage)
features.outputFeatures.link(features_out.input)
#detection.out.link(detection_out.input)
#imu.out.link(imu_out.input)

streams = [RIGHT_STREAM_NAME, DEPTH_RIGHT_STREAM_NAME, FEATURES_STREAM_NAME]
cvColorMap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
cvColorMap[0] = [0, 0, 0]
print("Creating DepthAI device")
device = dai.Device()
def drawFeatures(frame, features):
    pointColor = (0, 0, 255)
    circleRadius = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    thickness = 2
    for idx, feature in enumerate(features):
        #print(f"{feature.position.x}:{feature.position.y}")
        #cv2.putText(frame, str(idx), (int(feature.position.x), int(feature.position.y)), font, fontScale, pointColor, thickness, cv2.LINE_AA)
        cv2.circle(frame, (int(feature.position.x), int(feature.position.y)), circleRadius, pointColor, -1, cv2.LINE_AA, 0)

edge = 20
depth_cutout_margin = 30
first = True
first_save = True
with device:
    device.startPipeline(pipeline)

    # Create a receive queue for each stream
    right_q = device.getOutputQueue(RIGHT_STREAM_NAME, 8, blocking=False)
    depth_q = device.getOutputQueue(DEPTH_RIGHT_STREAM_NAME, 8, blocking=False)
    feature_q = device.getOutputQueue(FEATURES_STREAM_NAME, 8, blocking=False)
    
    while True:
	    
        right_frame = right_q.get().getCvFrame()
        cv2.imshow(RIGHT_STREAM_NAME, right_frame)

        depth_frame = depth_q.get().getCvFrame()
        selected_depth = depth_frame[200-depth_cutout_margin:200+depth_cutout_margin, 320-depth_cutout_margin:320+depth_cutout_margin]
        
        average_depth = np.nanmedian(selected_depth[selected_depth!=0])*1000/parameters.depth_unit_multiplier
        print(f"{average_depth}")
        depth_frame = (depth_frame*(50*1000/parameters.depth_unit_multiplier)).astype(np.uint16)
        depth_frame = getDisparityFrame(depth_frame, cvColorMap)
        cutout = depth_frame[200-depth_cutout_margin:200+depth_cutout_margin, 320-depth_cutout_margin:320+depth_cutout_margin]
        cv2.imshow(DEPTH_RIGHT_STREAM_NAME, cutout)
        tracked_features = feature_q.get().trackedFeatures
        tracked_features = [features for features in tracked_features if (features.position.x > edge and features.position.x < 640-edge and features.position.y > edge and features.position.y < 400 - edge)]
        #print(f"{len(tracked_features)}")
        drawFeatures(depth_frame, tracked_features)
        cv2.imshow(FEATURES_STREAM_NAME, depth_frame)

        if first:
            cv2.moveWindow(RIGHT_STREAM_NAME, 200, 300) 
            cv2.moveWindow(DEPTH_RIGHT_STREAM_NAME, 600, 300) 
            cv2.moveWindow(FEATURES_STREAM_NAME, 1000, 300) 
            first = False

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord("s"):
            print("saving")
            folder = f"{save_folder}"
            if first_save:
                os.makedirs(folder)
                first_save = False
            cv2.imwrite(f"{folder}/{name}_{save_count}_greyscale_{average_depth:.1f}.jpg", right_frame)
            cv2.imwrite(f"{folder}/{name}_{save_count}_cutout_{average_depth:.1f}.jpg", cutout)
            cv2.imwrite(f"{folder}/{name}_{save_count}_depth_{average_depth:.1f}.jpg", depth_frame)
            save_count+=1
