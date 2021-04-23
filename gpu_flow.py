#!/usr/bin/python3

# Imports

import os
import argparse
import time

#import tqdm
import cv2
import numpy as np

# Functions

def to_img(raw_flow, min_bound, max_bound,
        pix_min, pix_max):
    """
    Convert raw flow data from min-max bound to pixel bounds.
    """

    # First clip extreme values
    clipped_flow = raw_flow.clip(min_bound, max_bound)

    # Then scale to [0, 1]
    zero_one_flow = np.true_divide(clipped_flow - min_bound, max_bound - min_bound)

    # And finally convert to new range
    pix_flow = zero_one_flow * (pix_max - pix_min) + pix_min

    return pix_flow.astype(np.uint8)


def read_video(video_file):
    """
    Read a video file from disk, returning cv2 video capture object.
    WARNING: the capture needs to be .release()'d in every possible
    code execution path.
    """
    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        print('Error opening video stream: {}' \
                .format(video_file))
        cap.release()
        raise IOError

    print('Opened video capture, total frames: {}' \
            .format(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    return cap

def save_flows(flow, save_path, frame_num, bounds, pixrange, xmark='x', ymark='y', filetype='png'):
        """
        Function that saves flow images to disk after postprocessing.
        """
        args = bounds + pixrange

        print('Flow shape: {}'.format(flow.shape))

        flow_img = to_img(flow, *args)

        #tarshape = list(flow_img[..., 0].shape)
        #tarshape.append(3)
        #target_x = np.zeros(tuple(tarshape))
        #target_y = np.zeros(tuple(tarshape))
        #for i in range(3):
        #    target_x[..., i] = flow_img[..., 0]
        #    target_y[..., i] = flow_img[..., 1]


        #target_x = to_img(flow[..., 0], *args)
        #target_y = to_img(flow[..., 1], *args)


        target_x = flow_img[..., 0]
        target_y = flow_img[..., 1]

        print('DEBUG flow images')
        print(type(target_x))
        print(target_x.shape)
        print(target_x.dtype)

        fname = 'flow_{}'+'_{:08d}.{}'.format(frame_num, filetype)
        print(os.path.join(save_path, fname.format('??')))
        #cv2.imwrite('temp.png', target_x)#, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        cv2.imwrite(os.path.join(save_path, fname.format(xmark)), target_x)#, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        cv2.imwrite(os.path.join(save_path, fname.format(ymark)), target_y)#, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

def comp_next_flow(video_capture, prev_gray, flow_comp, \
        frame_num, bounds, pixrange, save_path, use_cuda, \
        filetype='png', save_img=False):
    """
    Get next optical flow frame, given video capture object at the
    current frame and the preious camera frame.
    """

    # Get next camera frame
    flag, new_frame = video_capture.read()

    # Check if the video is over
    if not flag:
        return None

        # check if the raw frames need to be saved
        if save_img:
            cv2.imwrite(new_frame, os.path.join(save_path, \
                    'frame_{:08d}.{}'.format(frame_num, filetype)))

    # Compute auxiliary variables
    gray_raw = cv2.cvtColor(new_frame, cv2.COLOR_RGB2GRAY)

    if (use_cuda):
        gray = cv2.cuda_GpuMat()
        #prev_gray = cv2.cuda_GpuMat()
        gray.upload(gray_raw)
        #prev_gray.upload(prev_gray)
    else:
        gray = gray_raw
    
    print('Computing flow!')
    # Compute flow
    flow_raw = flow_comp(prev_gray, gray, None)
    
    if (use_cuda):
        flow = flow_raw.download()
    else:
        flow = flow_raw

    #save_flows(flow.image, save_path, frame_num, bounds, pixrange)
    save_flows(flow, save_path, frame_num, bounds, pixrange)

    return gray

def skip_frames(video_capture, num_frames):
    """
    Advance the video_capture object by num_frames.
    Return number of frames passed over.
    """
    ret = 0
    for t in range(num_frames):
        flag, _ = video_capture.read()
        if flag:
            ret += 1
    return ret

def flowify(video_file, step, bounds, save_path, no_cuda, pixrange=[0,255]):
    """
    Extract flow images from an entire video.
    """

    # Start the video reading
    cap = read_video(video_file)

    try:
        cuda_avail = cv2.cuda.getCudaEnabledDeviceCount() > 0
    except:
        cuda_avail = False

    use_cuda = cuda_avail and (not no_cuda)

    loop_times = [] # keep some perf stats
    try:
        # Set up flow computation
        #flowDTLV1 = cv2.createOptFlow_DualTVL1()
        #flowDTLV1 = cv2.DualTVL1OpticalFlow_create()
        if (use_cuda):
            print('Setting CUDA optical flow computation...')
            flowDTLV1 = cv2.cuda_OpticalFlowDual_TVL1.create()
        else:
            print('Setting CPU opticla flow computation...')
            flowDTLV1 = cv2.optflow.DualTVL1OpticalFlow_create()
        flow_comp = flowDTLV1.calc

        # Bootstrap with the first frame
        _, first_frame = cap.read()
        first_gray_raw = cv2.cvtColor(first_frame, cv2.COLOR_RGB2GRAY)
        if use_cuda:
            first_gray = cv2.cuda_GpuMat()
            first_gray.upload(first_gray_raw)
        else:
            first_gray = first_gray_raw

        frame_num = 1

        # Start and loop
        frame_num += skip_frames(cap, step-1)
        gray = comp_next_flow(cap, first_gray, flow_comp, \
                frame_num, bounds, pixrange, save_path, use_cuda)
        frame_num += 1
        while (gray is not None):
            start = time.time()
            frame_num += skip_frames(cap, step-1)
            print('DEBUG frame number')
            print('expected: {}, actual: {}'.format(frame_num, \
                    cap.get(cv2.CAP_PROP_POS_FRAMES)))
            new_gray = comp_next_flow(cap, gray, flow_comp, \
                    frame_num, bounds, pixrange, save_path, use_cuda)
            gray = new_gray
            frame_num += 1
            end = time.time()
            loop_times.append(end-start)
    finally:
        # always release video resource
        cap.release()
        print('Looping avg/std: {}, {}' \
                .format(np.mean(loop_times), \
                np.std(loop_times)))

def get_video_list(root_path, recursive=False, \
        valid_exts=['.mp4', '.avi']):
    """
    Get list of files to parse from a root directory.
    """

    video_list = []
    for sub in os.listdir(root_path):
        sub_path = os.path.join(root_path, sub)
        if (os.path.isdir(sub_path)):
            if recursive:
                video_list.append(get_video_list(sub_path))
        elif (os.path.splitext(sub_path)[-1] in valid_exts):
            video_list.append(sub_path)

    return video_list

def execute(vids, step, bound, serial, save_path, no_cuda):
    """
    Execution manager for extracting flow from a batch of videos.
    If serial is False, multiprocessing is used, else videos are
        processed one at a time.
    """
    if serial:
        for v in vids:
            flowify(v, step, [-bound, bound], save_path, no_cuda)
    else:
        raise NotImplementedError


# Driver code

def main(args):

    assert not (args.no_cuda and args.cuda_device), \
            'Impossible input arguments configuration'

    if args.cuda_device:
        cv2.cuda.setDevice(args.cuda_device)

    print('Finding videos in: {}'.format(args.data_root))
    vl = get_video_list(args.data_root, args.recursive)
    if (args.verbose):
        print(vl)
    print('Done, found {} in total.'.format(len(vl)))

    print('Executing flow computations...')
    execute(vl, args.frame_step, args.bound, args.serial, \
           args.output_dir, args.no_cuda)
    print('Done, results in: {}'.format(args.output_dir))


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='''
        Extract dense optical flow from video files.
    ''')

    parser.add_argument('data_root', help='root directory to look for data')
    parser.add_argument('--recursive', '-r', action='store_true', help='add this switch to recursively search data_root for files, otherwise only looks at the direct contents of data_root only')
    parser.add_argument('output_dir', help='output directory')
    parser.add_argument('--serial', '-s', action='store_true', help='add this switch to disable multiprocessing')
    parser.add_argument('--frame_step', '-f', type=int, default=1, help='determines the number of frames between flow computation data (default: 1)')
    parser.add_argument('--bound', '-b', type=int, default=20, help='the absolute bound on fow magnitude, i.e. flow will be in: [-bound, bound] (default: 20)')
    parser.add_argument('--verbose', '-v', action='store_true', help='add this switch for extra messages during execution')
    parser.add_argument('--no_cuda', '-n', action='store_true', help='add this switch to force CPU execution')
    parser.add_argument('--cuda_device', '-c', type=int, help='choose specific device to run on')

    parser.set_defaults(serial=False, recursive=False, verbose=False, no_cuda=False)

    args = parser.parse_args()

    main(args)

