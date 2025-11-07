#python3
#This script contains function to create NiPype workflow for analysing HCP dataset with SPM.

from nipype.interfaces import spm
matlab_cmd = '/opt/spm12-r7771/run_spm12.sh /opt/matlabmcr-2010a/v713/ script'
spm.SPMCommand.set_mlab_paths(matlab_cmd=matlab_cmd, use_mcr=True)

from nipype.interfaces.spm import (Coregister, Smooth, OneSampleTTestDesign, EstimateModel, EstimateContrast, 
                                   Level1Design, TwoSampleTTestDesign, Realign, 
                                   Normalize12, NewSegment)
from nipype.interfaces.fsl import ExtractROI, Info
from nipype.interfaces.spm import Threshold 
from nipype.algorithms.modelgen import SpecifySPMModel
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.algorithms.misc import Gunzip
from nipype import Workflow, Node, MapNode, JoinNode
from nipype.interfaces.base import Bunch

from os.path import join as opj
import os
import json

def get_preprocessing(exp_dir, result_dir, working_dir, output_dir, subject_list, task_list, fwhm_list):
    """
    Returns the SPM preprocessing workflow.
    Parameters: 
        - exp_dir: str, directory where raw data are stored
        - result_dir: str, directory where results will be stored
        - working_dir: str, name of the sub-directory for intermediate results
        - output_dir: str, name of the sub-directory for final results
        - subject_list: list of str, list of subject for which you want to do the preprocessing
        - task_list: list of str, list of task for which you want to do the preprocessing
        - fwhm_list: list of int, list of fhwm kernel for smoothing 
        
    Returns: 
        - preprocessing: Nipype WorkFlow 
    """
    # Iterables
    infosource_preproc = Node(IdentityInterface(fields = ['subject_id', 'task', 'fwhm']), 
        name = 'infosource_preproc')

    infosource_preproc.iterables = [('subject_id', subject_list), ('task', task_list), ('fwhm', fwhm_list)]

    # Templates to select files node
    anat_file = opj('STRUCTURAL', '{subject_id}', 'unprocessed', '3T', 'T1w_MPR1', 
                    '{subject_id}_3T_T1w_MPR1.nii.gz')

    func_file = opj('{task}', '{subject_id}', 'unprocessed', '3T', 'tfMRI_{task}_LR', 
                    '{subject_id}_3T_tfMRI_{task}_LR.nii.gz')

    template = {'anat' : anat_file, 'func' : func_file}

    # SelectFiles node - to select necessary files
    selectfiles_preproc = Node(SelectFiles(template, base_directory=exp_dir), name = 'selectfiles_preproc')

    # GUNZIP NODE : SPM do not use .nii.gz files
    gunzip_anat = Node(Gunzip(), name = 'gunzip_anat')

    gunzip_func = Node(Gunzip(), name = 'gunzip_func')

    ## Motion correction
    realign = Node(Realign(register_to_mean = True), 
                                name = 'realign')
    ## Coregistration 
    coregistration = Node(Coregister(jobtype = 'estimate', 
                           cost_function = 'nmi', 
                           fwhm = [7, 7], 
                           separation = [4, 2], 
                           tolerance=[0.02, 0.02, 0.02, 0.001, 0.001, 0.001, 0.01, 0.01, 
                                      0.01, 0.001, 0.001, 0.001]), name = 'coregistration')

    ## Segmentation  
    tissue1 = [('/opt/spm12-r7771/spm12_mcr/spm12/tpm/TPM.nii', 1), 1, (True,False), (False, False)]
    tissue2 = [('/opt/spm12-r7771/spm12_mcr/spm12/tpm/TPM.nii', 2), 1, (True,False), (False, False)]
    tissue3 = [('/opt/spm12-r7771/spm12_mcr/spm12/tpm/TPM.nii', 3), 2, (True,False), (False, False)]
    tissue4 = [('/opt/spm12-r7771/spm12_mcr/spm12/tpm/TPM.nii', 4), 3, (True,False), (False, False)]
    tissue5 = [('/opt/spm12-r7771/spm12_mcr/spm12/tpm/TPM.nii', 5), 4, (True,False), (False, False)]
    tissue6 = [('/opt/spm12-r7771/spm12_mcr/spm12/tpm/TPM.nii', 6), 2, (False,False), (False, False)]
    tissue_list = [tissue1, tissue2, tissue3, tissue4, tissue5, tissue6]
    
    segment = Node(NewSegment(write_deformation_fields = [True, True], tissues = tissue_list, 
                         channel_info = (0.001, 60, (True, True)), 
                         warping_regularization = [0, 0.001, 0.5, 0.05, 0.2],
                         affine_regularization = 'mni', 
                         sampling_distance = 3), name = 'segment')
    
    ## Normalization
    normalize_functional = Node(Normalize12(jobtype = 'write', write_voxel_sizes = [2, 2, 2],
                                write_interp = 4, write_bounding_box = [[-78, -112, -70], 
                                                                        [78, 76, 85]]),
                     name = 'normalize_functional')

    ## Smoothing
    smooth = Node(Smooth(implicit_masking = False), name = 'smooth')

    # DataSink Node - store the wanted results in the wanted repository
    datasink_preproc = Node(DataSink(base_directory=result_dir, container=output_dir), name='datasink_preproc')

    # Nipype preprocessing Workflow
    preprocessing =  Workflow(base_dir = opj(result_dir, working_dir), name = "preprocess_spm")
    # Connections between Nodes
    preprocessing.connect([(infosource_preproc, selectfiles_preproc, [('subject_id', 'subject_id'),
                                                                     ('task', 'task')]), 
                            (infosource_preproc, smooth, [('fwhm', 'fwhm')]),
                           (selectfiles_preproc, gunzip_func, [('func', 'in_file')]),
                           (selectfiles_preproc, gunzip_anat, [('anat', 'in_file')]),
                           (gunzip_func, realign, [('out_file', 'in_files')]),
                           (realign, coregistration, [('mean_image', 'source'), 
                                                       ('realigned_files', 'apply_to_files')]),
                           (gunzip_anat, coregistration, [('out_file', 'target')]),
                           (gunzip_anat, segment, [('out_file', 'channel_files')]),
                           (coregistration, normalize_functional, [('coregistered_files', 'apply_to_files')]),
                           (segment, normalize_functional, [('forward_deformation_field', 'deformation_file')]),
                           (normalize_functional, smooth, [('normalized_files', 'in_files')]),
                           (realign, datasink_preproc, 
                            [('realignment_parameters', 'preprocess_spm.@parameters')]),
                           (smooth, datasink_preproc, [('smoothed_files', 'preprocess_spm.@smooth')]),
                           (segment, datasink_preproc, [('native_class_images', 'preprocess_spm.@seg_maps_native'),
                                                    ('normalized_class_images', 'preprocess_spm.@seg_maps_norm')])])
    
    return preprocessing  


def get_subject_infos(event_file, contrasts):
    '''
    Create Bunchs for specifyModel.
    Parameters :
    - event_file: list of str, event files for the subject
    - contrasts: list of contrast to analyze
    
    Returns :
    - subject_info : Bunch for 1st level analysis.
    '''
    from nipype.interfaces.base import Bunch
    import numpy as np
    
    # Global lists 
    cond_names = sorted(contrasts) 
    onsets = []
    durations = []

    print(sorted(contrasts))

    # Selection of only event files corresponding to selected contrasts
    event_files = [f for f in event_file if f.split('/')[-1].split('.')[0] in contrasts]

    print(event_files)

    for i, c in enumerate(sorted(contrasts)):
        onset = [] # For each contrast, save the onset and duration
        duration = []
        file = sorted(event_files)[i]
        with open(file, 'rt') as f:
            for line in f:
                info = line.strip().split()
                onset.append(float(info[0])) 
                duration.append(float(info[1]))
        onsets.append(onset)
        durations.append(duration) # Add these onset and duration to global list 

    subject_info = Bunch(conditions=cond_names,
                             onsets=onsets, # Onsets and Durations contain list of lists with onsets and durations for every contrast
                             durations=durations,
                             amplitudes=None,
                             regressor_names=None,
                             regressors=None)

    return subject_info

def get_24_param(param_file):
    # Function to apply bash script on parameter file to obtain 24 MC parameters from the 6 firsts
    import os 

    out_file = os.path.join(os.path.dirname(param_file), '24_' + os.path.basename(param_file))

    os.system(f'bash /srv/tempdd/egermani/hcp_pipelines/src/lib/mp_diffpow24.sh {param_file} {out_file}')

    return out_file

def get_l1_analysis(exp_dir, output_dir, working_dir, result_dir, subject_list, task_list, contrast_list, fwhm_list, nb_param, hrf):
    """
    Returns the SPM first level analysis workflow.
    Parameters: 
        - exp_dir: str, directory where raw data are stored
        - result_dir: str, directory where results will be stored
        - working_dir: str, name of the sub-directory for intermediate results
        - output_dir: str, name of the sub-directory for final results
        - subject_list: list of str, list of subject for which you want to do the analysis
        - task_list: list of str, list of task for which you want to do the analysis
        - contrast_list: list of str, list of contrast for which you want to do the analysis
        - fwhm_list: list of int, list of fwhm kernel for smoothing 
        - nb_param: list of int, list of number of parameters for model -> one of 0, 6 or 24
        - hrf: list of string, derivatives for hrf function (derivatives or no_derivatives)

    Returns: 
        - l1_analysis : Nipype WorkFlow 
    """
    # Infosource Node - To iterate on subjects and other parameters
    infosource = Node(IdentityInterface(fields = ['subject_id', 'task', 'fwhm', 'nb_param', 'hrf']),
                      name = 'infosource')

    infosource.iterables = [('subject_id', subject_list), ('task', task_list), 
                            ('fwhm', fwhm_list), ('nb_param', nb_param), ('hrf', hrf)]

    # Templates to select files node
    param_file = opj(output_dir, 'preprocess_spm', '_fwhm_{fwhm}_subject_id_{subject_id}_task_{task}',
                    'rp_{subject_id}_3T_tfMRI_{task}_LR.txt')

    func_file = opj(output_dir, 'preprocess_spm', '_fwhm_{fwhm}_subject_id_{subject_id}_task_{task}',
                'swr{subject_id}_3T_tfMRI_{task}_LR.nii')

    event_file = opj(exp_dir, '{task}', '{subject_id}', 'unprocessed', '3T', 'tfMRI_{task}_LR', 'LINKED_DATA', 'EPRIME', 'EVs', '*.txt')

    template = {'param' : param_file, 'func' : func_file, 'event' : event_file}

    # SelectFiles node - to select necessary files
    selectfiles = Node(SelectFiles(template, base_directory=result_dir), name = 'selectfiles')
    
    # DataSink Node - store the wanted results in the wanted repository
    datasink = Node(DataSink(base_directory=result_dir, container=output_dir), name='datasink')

    # Get Subject Info - get subject specific condition information
    subject_infos = Node(Function(input_names=['event_file', 'contrasts'],
                                   output_names=['subject_info'],
                                   function=get_subject_infos),
                          name='subject_infos')

    subject_infos.inputs.contrasts = contrast_list

    # SpecifyModel - Generates SPM-specific Model
    specify_model = Node(SpecifySPMModel(input_units = 'secs', output_units = 'secs',
                                        time_repetition = 0.72, high_pass_filter_cutoff = 128), name='specify_model')

    # Level1Design - Generates an SPM design matrix
    if hrf == ['derivatives']: # Use derivatives of the HRF 
        hrf_values = [1, 1]
    else:
        hrf_values = [0, 0]

    l1_design = Node(Level1Design(bases = {'hrf': {'derivs': hrf_values}}, timing_units = 'secs', 
                                    interscan_interval = 0.72), name='l1_design')

    # EstimateModel - estimate the parameters of the model
    l1_estimate = Node(EstimateModel(estimation_method={'Classical': 1}),
                          name="l1_estimate")

    # EstimateContrast - estimates contrasts
    contrast_estimate = Node(EstimateContrast(), name="contrast_estimate")
    contrast_estimate.inputs.contrasts = [(f'{c} vs baseline', 'T', [c], [1]) for c in sorted(contrast_list)]

    # Create l1 analysis workflow and connect its nodes
    l1_analysis = Workflow(base_dir = opj(result_dir, working_dir), name = "l1_analysis")

    l1_analysis.connect([(infosource, selectfiles, [('subject_id', 'subject_id'), ('task', 'task'), ('fwhm', 'fwhm')]),
                        (subject_infos, specify_model, [('subject_info', 'subject_info')]),
                        (selectfiles, subject_infos, [('event', 'event_file')]),
                        (selectfiles, specify_model, [('func', 'functional_runs')]),
                        (specify_model, l1_design, [('session_info', 'session_info')]),
                        (l1_design, l1_estimate, [('spm_mat_file', 'spm_mat_file')]),
                        (l1_estimate, contrast_estimate, [('spm_mat_file', 'spm_mat_file'),
                                                          ('beta_images', 'beta_images'),
                                                          ('residual_image', 'residual_image')]),
                        (contrast_estimate, datasink, [('con_images', 'l1_analysis_spm.@con_images'),
                                                                ('spmT_images', 'l1_analysis_spm.@spmT_images'),
                                                                ('spm_mat_file', 'l1_analysis_spm.@spm_mat_file')]),
                        ])

    if nb_param == [6]: # If only 6 MC parameters to use, use file output from preprocessing workflow without modification

        l1_analysis.connect([(selectfiles, specify_model, [('param', 'realignment_parameters')])])

    elif nb_param == [24]: # If use of 24 MC parameters, need to apply a bash script on the original parameter file before to obtain the 24 MC param

        param_extent = Node(Function(input_names=['param_file'],
                                   output_names=['out_file'],
                                   function=get_24_param),
                          name='param_extent')

        l1_analysis.connect([(selectfiles, param_extent, [('param', 'param_file')]), 
            (param_extent, specify_model, [('out_file', 'realignment_parameters')])])
        
    # If use of 0 MC parameters, no connections with this file.

    return l1_analysis