
Acoustic parametrization
========================

This notebook illustrate the basic of acoustic parametrization using
**SIDEKIT**. Acoustic parametrization is performed by the ``frontend``
module in **SIDEKIT**, see the API documentation of this module for more
information.

This notebook is organized in 2 parts: - a description of the
``FeaturesServer``: an object that provide a high level interface to the
frontend module - a description of the main low level functions that can
be used to extract, select and normalize acoustic parameters from an
audio signal

1. FeaturesServer
-----------------

The ``FeatureServer`` provides a high level interface to acoustic
parameters in **SIDEKIT**. Its purpose is to process audio files or
parameters already extracted with **SIDEKIT** or another compatible tool
and to feed them to other **SIDEKIT**'s objects.

A feature server is used in 2 steps: - initialization of the server
(``where you define your configuration``) - processing of the data

In this second step, **SIDEKIT** can: - process the audio signal and
store the acoustic parameters to disk - process the audio signal and
transfer the parameters to another part of **SIDEKIT** (to avoid
storage) - read stored acoustic parameters and transfer those parameters
to another part of **SIDEKIT** (to avoid storage)

1.1 Initialization of the FeaturesServer and options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Below is the list of options available for the ``FeaturesServer``
together with their default value.

+------------------------------+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Option                       | Default value            | Description                                                                                                                                                                                 |
+==============================+==========================+=============================================================================================================================================================================================+
| input\_dir                   | ``'./'``                 | directory where input data are stored                                                                                                                                                       |
+------------------------------+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| input\_file\_extension       | ``'.wav'``               | extension of the input files (including **.**)                                                                                                                                              |
+------------------------------+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| label\_dir                   | ``'./'``                 | directory where input labels are stored if used                                                                                                                                             |
+------------------------------+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| label\_file\_extension       | ``'.lbl'``               | extension of the input label files (including **.**)                                                                                                                                        |
+------------------------------+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| from\_file                   | ``'audio'``              | type of file to read, could be 'audio'                                                                                                                                                      |
+------------------------------+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| config                       | None                     | enable one of the pre-defined configurations (``'diar_16k'``, ``'diar_8k'``, ``sid_8k``, ``sid_16k``, ``fb_8k``) manually define options replace the pre-defined ones                       |
+------------------------------+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| single\_channel\_extension   | ``('')``                 | if the audio input is single channel, add this extension to the filename                                                                                                                    |
+------------------------------+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| double\_channel\_extension   | ``('_a', '_b')``         | if the audio input is double channel, add the first and second extension to the respective channel filename                                                                                 |
+------------------------------+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| sampling\_frequency          | 8000                     | sampling frequency of the input                                                                                                                                                             |
+------------------------------+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| lower\_frequency             | 0                        | lower frequency of the lower filter in the filter-bank                                                                                                                                      |
+------------------------------+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| higher\_frequency            | sampling\_frequency/2.   | higher frequency of the lower filter in the filter-bank                                                                                                                                     |
+------------------------------+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| linear\_filters              | 0                        | number of linear filters if using a linear scale                                                                                                                                            |
+------------------------------+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| log\_filters                 | 40                       | number of log filters if using a linear scale                                                                                                                                               |
+------------------------------+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| window\_size                 | 0.025                    | duration of the sliding window in seconds                                                                                                                                                   |
+------------------------------+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| shift                        | 0.01                     | shift of the sliding window in seconds                                                                                                                                                      |
+------------------------------+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ceps\_number                 | 13                       | number of DCT coefficient to keep (final number of cepstral coefficients)                                                                                                                   |
+------------------------------+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| vad                          | None                     | type of voice activity detection, can be ``'energy'`` to use a 3-Gaussian energy-based VAD, ``'snr'`` to use an energy-based VAD based on an SNR estimation or None                         |
+------------------------------+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| snr                          | 40                       | for ``'snr'`` vad, fix the reference level of the SNR                                                                                                                                       |
+------------------------------+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| feat\_norm                   | None                     | feature normalization applied, can be ``'cms'`` for cepstral mean subtraction, ``'cmvn'`` for cepstral mean variance normalization, ``'stg'`` for short term Gaussianization or None        |
+------------------------------+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| log\_e                       | False                    | boolean, if True stack the log-energy coefficient in first position                                                                                                                         |
+------------------------------+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| rasta                        | False                    | boolean, if True, apply RASTA filtering                                                                                                                                                     |
+------------------------------+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| delta                        | False                    | boolean, if True, add the first order derivative to the cepstral coefficients                                                                                                               |
+------------------------------+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| double\_delta                | False                    | boolean, if True, add the first order derivative to the cepstral coefficients                                                                                                               |
+------------------------------+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| delta\_filter                | None                     | if None, the derivative are computed between two points; if a numpy ndarray of coefficients is provided, those coefficients are used(for instance: ``[.25, .5, .25, 0, -.25, -.5, -.25]``   |
+------------------------------+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| mask                         | None                     | keep only the MFCC index present in the mask list                                                                                                                                           |
+------------------------------+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| dct\_pca                     | False                    | boolean, if True, add temporal context to the frames by using a temporal DCT                                                                                                                |
+------------------------------+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| dct\_pca\_config             | (12, 12, None)           | tuple of three values, the left and right context defining the window to compute the DCT and a PCA matrix if already computed                                                               |
+------------------------------+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| sdc                          | False                    | boolean, if True, temporal contextualization includes shifted delta cepstral coefficients                                                                                                   |
+------------------------------+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| sdc\_config                  | (1, 3, 7)                | tuple of 3 values that define the configuration of the SDC coefficients following the standard definition                                                                                   |
+------------------------------+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| keep\_all\_features          | False                    | boolean, if True keep all selected and non-selected frames. If False, onnly selected frames are returned (useful to save space on the disk)                                                 |
+------------------------------+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| spec                         | False                    | boolean, if True returns the spectrogram                                                                                                                                                    |
+------------------------------+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| mspec                        | False                    | boolean, if True; return the filter-bank coefficients (inplace of the cepstral coefficients).                                                                                               |
+------------------------------+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

1.2 Example: extract MFCC parameters and store them in SPRO4 format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following configuration is the one used in NIST-SRE 2010 tutorials
and provides decent results on this task.

.. code:: python

    import scipy
    import sidekit
    import matplotlib
    import numpy as np
    import matplotlib.pyplot as plt
    %matplotlib inline  

.. code:: python

    fs = sidekit.FeaturesServer(input_dir='./sph/',
                     input_file_extension='.sph',
                     label_dir='./lbl/',
                     label_file_extension='.lbl',
                     from_file='audio',
                     config=None,
                     single_channel_extension=(''),
                     double_channel_extension=('_a', '_b'),
                     sampling_frequency=8000,
                     lower_frequency=200,
                     higher_frequency=3800,
                     linear_filters=0,
                     log_filters=24,
                     window_size=0.025,
                     shift=0.01,
                     ceps_number=13,
                     snr=40,
                     vad='snr',
                     feat_norm='cmvn',
                     log_e=True,
                     dct_pca=False,
                     dct_pca_config=None,
                     sdc=False,
                     sdc_config=None,
                     delta=True,
                     double_delta=True,
                     delta_filter=np.array([.25, .5, .25, 0, -.25, -.5, -.25]),
                     rasta=True,
                     keep_all_features=True,
                     spec=False,
                     mspec=False)

Once initialized, we create - a list of audio files to process - a list
of output feature files.

and then extract the features.

Note that the number of audio channels (1 or 2) is managed by the
FeaturesServer itself. One unique filename is provided per audio file
(even stereo files).

.. code:: python

    import glob
    audio_file_list = [ff.split('/')[-1].split('.')[0] for ff in glob.glob("sph/*.sph")]
    feature_file_list = [ff.split('/')[-1].split('.')[0] for ff in audio_file_list]
    
    print("audio_file_list[:3] = {}".format(audio_file_list[:3]))
    print("feature_file_list[:3] = {}".format(feature_file_list[:3]))

.. code:: python

    fs.save_list(audio_file_list=audio_file_list[:3], 
                 feature_file_list=feature_file_list[:3], 
                 mfcc_format='spro4', 
                 feature_dir='./feat/', 
                 feature_file_extension=".mfcc",
                 and_label=True, 
                 numThread=2)

Note that when using the ``save_list`` function, label files are saved
exactly in the same directory as the feature files.

.. code:: python

    import glob
    mfcc_file_list = glob.glob("feat/*.mfcc")
    label_file_list = glob.glob("feat/*.lbl")
    
    print("mfcc_file_list[:3] = {}".format(mfcc_file_list[:3]))
    print("label_file_list[:3] = {}".format(label_file_list[:3]))

The previous code read audio files in SPHERE format, extract 13 cepstral
coefficient plus the log energy and their first and second derivatives.
RASTA filtering and cmvn are applied. In this example, all files are
mono-channel so, for the first files, we get:

-  ``'./mfcc/xaaf.mfcc'``
-  ``'./mfcc/xaag.mfcc'``
-  ``'./mfcc/xaao.mfcc'``

Those files include all frames (speech and non-speech) and additional
label files are created as: - ``'./lbl/xaaf.lbl'`` -
``'./lbl/xaag.lbl'`` - ``'./lbl/xaao.lbl'``

2. Parametrization via low level functions
------------------------------------------

In case you can't or you don't want to use a FeaturesServer, you can
easily create your own front-end using **SIDEKIT** low level functions.

.. code:: python

    audio_filename = "./sph/tfbax.sph"
    sampling_frequency = 8000

Read the audio signal from a SPHERE file and plot the signal from the
first channel.

.. code:: python

    x, rate = sidekit.frontend.io.read_audio(audio_filename, sampling_frequency)
    plt.subplot(2,1,1)
    plt.plot(x[:, 0])
    plt.subplot(2,1,2)
    plt.plot(x[:, 1])




.. parsed-literal::

    [<matplotlib.lines.Line2D at 0x10277f400>]




.. image:: output_15_1.png


Note that x is a N x c ndarray where N is the number of samples and c is
the number of channels (1 for mono, 2 for stereo).

After loading the signal, we usually add it a small-amplitude random
noise in order to avoid issues due to zeros.

.. code:: python

    np.random.seed(0)  # Initialize the random seed
    x[:, 0] += 0.0001 * np.random.randn(x.shape[0])  # add a small random noise to avoid numerical issues
    plt.plot(x[:, 0])




.. parsed-literal::

    [<matplotlib.lines.Line2D at 0x1053576a0>]




.. image:: output_17_1.png


Note that in the following tutorial, we will only process the first
channel of a stereo file.

In practice, you should apply the same treatment to both channels.

If the number of samples is not enough to extract a single frame, you
should not process the file.

In the feature server high level function, we return a zero-length
sequence of frames and zero-length label vector with the right frame
dimension to avoid cascade issues.

.. code:: python

    window_size = 0.025  # length of the sliding window given in seconds

.. code:: python

    if x.shape[0] < sampling_frequency * window_size:
        print("Not enough data to process")
        cep_size = self.ceps_number * (1 + int(self.delta) + int(self.double_delta))\
                   + int(self.mspec) * (self.linear_filters + self.log_filters)
        cep = np.empty((0, cep_size))
        label = np.empty((0, 1))
    else:
        print("Enough data to extract at least one frame, we can continue this tutorial.")


.. parsed-literal::

    Enough data to extract at least one frame, we can continue this tutorial.


.. code:: python

    x = x[:, 0]
    x = x[:, np.newaxis]

Now we compute the Cepstral coefficients, here 13 MFCCs.

The input parameters of this function provide all necessary options to
describe your configuration: - input\_sig: input signal, a one
dimensional ndarray (mandatory parameter, no default value) - lowfreq:
lower limit of the frequency band filtered (default is 100 Hz) -
maxfreq: higher limit of the frequency band filtered (default is 8000
Hz) - nlinfilt: number of linear filters to use (default is 0 linear
filters) - nlogfilt: number of log-linear filters to use (default is 24
mel filters) - nwin: length of the sliding window in seconds (default is
0.025 second) - fs: sampling frequency of the original signal (default
is 16,000 Hz) - nceps: number of cepstral coefficients to extract
(default is 13) - shift: shift between two analyses (default is 0.01
second) - get\_spec: boolean, if true returns the spectrogram (default
is False) - get\_mspec: boolean, if true returns the output of the
filter banks (default is False)

.. code:: python

    c = sidekit.frontend.features.mfcc(input_sig=x, 
                                       fs=sampling_frequency,
                                       lowfreq=200,
                                       maxfreq=3800,
                                       nlinfilt=0,
                                       nlogfilt=24,
                                       nwin=0.025, 
                                       nceps=13,
                                       get_spec=False, 
                                       get_mspec=False)

The output of the ``mfcc`` function is a list of 4 outputs: - the
cepstral coefficients (nceps-dimensional ndarray) - the log-energy
vector (1-dimensional ndarray) - the spectrogramm - the filter-bank
outputs

Here we plot the 3rd cepstral coefficient and the log-energy on the
first 500 frames.

.. code:: python

    plt.subplot(2,1,1)
    plt.plot(c[0][:500, 2])
    plt.subplot(2,1,2)
    plt.plot(c[1][:500])




.. parsed-literal::

    [<matplotlib.lines.Line2D at 0x104e7e6a0>]




.. image:: output_26_1.png


Note that in the present case, the third and fourth elements of the list
"c", output of the ``mfcc`` function are None.

You can ask to output all of the elements.

In case you are not interested into Cepstral coefficients but only in
filter-banks, you can set the ``ceps`` parameter to 0 and ``mspec`` to
True.

Example: we extract filter-banks and plot the output from the 3rd filter
(500 first frames).

.. code:: python

    fb = sidekit.frontend.features.mfcc(input_sig=x, 
                                       fs=sampling_frequency,
                                       lowfreq=200,
                                       maxfreq=3800,
                                       nlinfilt=0,
                                       nlogfilt=24,
                                       nwin=0.025, 
                                       nceps=0,
                                       get_spec=False, 
                                       get_mspec=True)
    plt.plot(fb[3][:500, 2])




.. parsed-literal::

    [<matplotlib.lines.Line2D at 0x109ec2da0>]




.. image:: output_28_1.png


Perform voice activity detection based on energy.

The output of the process is a ``label`` vector: vector of boolean. -
True = speech - False = non-speech

.. code:: python

    # VAD based on SNR
    window_sample = int(window_size * sampling_frequency)
    label_snr = sidekit.frontend.vad.vad_snr(x, 40, fs=sampling_frequency, shift=0.01, nwin=window_sample)
    
    # VAD based on 3 gaussian model
    label_3g = sidekit.frontend.vad.vad_energy(c[1], distribNb=3, nbTrainIt=8, flooring=0.0001, ceiling=1.5, alpha=0.1)

Example of label visualization.

We plot on the same graph two seconds of signal and the corresponding
speech labels (times 3000 for visualization sake).

.. code:: python

    signal_time = np.arange(x.shape[0])/8000
    label_time = np.arange(label_snr.shape[0])/100
    
    # plot between 112 and 114 seconds
    signal_box = (112 <= signal_time) & (signal_time <= 114)
    plt.plot(signal_time[signal_box], x[signal_box,0])
    
    label_box = (112 <= label_time) & (label_time <= 114)
    plt.plot(label_time[label_box], 3000*label_snr[label_box], color='r')
    
    plt.axis([112, 114, -3000, 4000])




.. parsed-literal::

    [112, 114, -3000, 4000]




.. image:: output_32_1.png


Add the log-energy as first coefficient

.. code:: python

    c[0] = np.hstack((c[1][:, np.newaxis], c[0]))

.. code:: python

    plt.subplot(2,1,1)
    plt.plot(c[0][:16000,0])
    plt.subplot(2,1,2)
    plt.plot(c[0][:16000,2])
    c[0].shape




.. parsed-literal::

    (30349, 14)




.. image:: output_35_1.png


Apply RASTA normalization

.. code:: python

    c[0] = sidekit.frontend.normfeat.rasta_filt(c[0])
    c[0][:2, :] = c[0][2, :]
    label_snr[:2] = label_snr[2]
    c[0][:10,:3]




.. parsed-literal::

    array([[ 0.        ,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        ],
           [ 0.85238597,  0.28377742,  0.04532976],
           [ 1.26571657,  0.27795799,  0.08213694],
           [ 1.26597738,  0.24695778,  0.06480024],
           [ 1.77677026,  0.20154712, -0.06302089],
           [ 1.74042667,  0.18127141, -0.09636657],
           [ 1.86793649,  0.20602675, -0.10437056]])



Add the first abd second order derivatives (delta and double delta).

The derivatives are computed using a filter.

Default is ``[.25, .5, .25, 0, -.25, -.5, -.25]``

.. code:: python

    delta = sidekit.frontend.features.compute_delta(c[0], 
                                            win=3, 
                                            method='filter', 
                                            filt=np.array([.25, .5, .25, 0, -.25, -.5, -.25]))
    cep = np.column_stack((c[0], delta))

And now add the double deltas

.. code:: python

    double_delta = sidekit.frontend.features.compute_delta(delta,
                                                           win=3, 
                                                           method='filter', 
                                                           filt=np.array([.25, .5, .25, 0, -.25, -.5, -.25]))
    cep = np.column_stack((cep, double_delta))

Smooth the labels by using morphological filtering.

This function takes a list of label vetors as input as it can also be
used to smooth both channels of a stereo file and remove all overlaping
parts of the detected speech.

.. code:: python

    label_snr = sidekit.frontend.vad.label_fusion(label_snr)

Speech labels before and after smoothing

.. code:: python

    cep[:10,:]
    mu = np.mean(cep[label_snr, :], axis=0)
    stdev = np.std(cep[label_snr, :], axis=0)

Now that we have the parameters, their temporal context and the speech
labels, we can use the selected speech frames in order to compute mean
and variance parameters to normalize using Cepstral Mean and Variance
Normalization (we could also use CMS or STG in the same manner).

.. code:: python

    sidekit.frontend.normfeat.cmvn(cep, label_snr)

.. code:: python

    print(cep.shape)
    print(cep[:10,:3])


.. parsed-literal::

    (30349, 42)
    [[-0.7734641  -0.21329845 -0.02642215]
     [-0.7734641  -0.21329845 -0.02642215]
     [-0.7734641  -0.21329845 -0.02642215]
     [-0.7734641  -0.21329845 -0.02642215]
     [-0.4209142   0.06694543  0.04121504]
     [-0.24995918  0.06119846  0.09613557]
     [-0.24985131  0.03058426  0.07026722]
     [-0.03858554 -0.01426096 -0.12045655]
     [-0.05361738 -0.0342842  -0.17021211]
     [-0.00087886 -0.00983711 -0.18215498]]

