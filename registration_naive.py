import os, sys, time
import argparse
import SimpleITK as sitk

def findDicomSeries(directory):
    sitk.ProcessObject_SetGlobalWarningDisplay(False)
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(directory)
    sitk.ProcessObject_SetGlobalWarningDisplay(True)
    return series_IDs

def imread(data_directory, series_ID):
    if isinstance(series_ID, list) or isinstance(series_ID, tuple):
        series_file_names = ()
        for id in series_ID:
            series_file_names+=sitk.ImageSeriesReader.GetGDCMSeriesFileNames(data_directory, id)
    else:
        series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(data_directory, series_ID)
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    series_reader.MetaDataDictionaryArrayUpdateOn()
    series_reader.LoadPrivateTagsOn()
    image3D = series_reader.Execute()
    return image3D, series_reader

class Registration:
    def __init__(self, ref_dir, tar_dir, ref_uid=None, tar_uid=None):
        refIDs = findDicomSeries(ref_dir)
        tarIDs = findDicomSeries(tar_dir)
        if ref_uid is None and refIDs:
            ref_uid = refIDs[0]
        if tar_uid is None and tarIDs:
            tar_uid = tarIDs[0]

        self.ref_img, self.ref_reader = imread(ref_dir, ref_uid)
        self.tar_img, self.tar_reader = imread(tar_dir, tar_uid)
            
        self.filenames = [os.path.basename(i) for i in self.ref_reader.GetFileNames()]
        self.ref_shape = self.ref_img.GetSize()
        self.tar_shape = self.tar_img.GetSize()
        self.reg_img = None
        self.transformation = self.__center_initialization__()
        self.reg_transform = None
    
    def __center_initialization__(self):
        initial_transform = sitk.Euler3DTransform(sitk.CenteredTransformInitializer(self.ref_img, 
                                                      self.tar_img, 
                                                      sitk.Euler3DTransform(), 
                                                      sitk.CenteredTransformInitializerFilter.GEOMETRY))
        
        return initial_transform
    
    def resampleOnly(self):
        identity = sitk.Transform(3, sitk.sitkIdentity)
        self.resampled_tar_img = sitk.Resample(
            sitk.Cast(self.tar_img, sitk.sitkFloat32), 
            sitk.Cast(self.ref_img, sitk.sitkFloat32), 
            identity, sitk.sitkLinear, 0.0, sitk.sitkInt16)

    def registration(self):
        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.10)
        registration_method.SetInterpolator(sitk.sitkLinear)
        registration_method.SetOptimizerAsGradientDescentLineSearch(learningRate=1.4,
                                                                numberOfIterations=100,
                                                                convergenceMinimumValue=1e-4,
                                                                convergenceWindowSize=5)
        registration_method.SetOptimizerScalesFromPhysicalShift()
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,2,1])
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        registration_method.SetInitialTransform(self.transformation, inPlace=False)
        self.reg_transform = registration_method.Execute(
            sitk.Cast(self.tar_img, sitk.sitkFloat32), 
            sitk.Cast(self.ref_img, sitk.sitkFloat32))
        self.reg_img = sitk.Resample(
            sitk.Cast(self.tar_img, sitk.sitkFloat32), 
            sitk.Cast(self.ref_img, sitk.sitkFloat32), 
            self.reg_transform, sitk.sitkLinear, 0.0, sitk.sitkInt16)
        
        self.transformation.SetCenter(self.reg_transform.GetFixedParameters()[:3])
        self.transformation.SetRotation(*self.reg_transform.GetParameters()[:3])
        self.transformation.SetTranslation(self.reg_transform.GetParameters()[3:])
        
    def save_transform(self, *args, **kwds):
        sitk.WriteTransform(self.transformation, *args, **kwds)

    def save_nifti(self):
        return

    def save_dicom(self, sitkImage, output_dir, new_series_n=999):
        os.makedirs(output_dir, exist_ok=True)
        writer = sitk.ImageFileWriter()
        writer.KeepOriginalImageUIDOn()

        tags_to_copy = ["0010|0010", # Patient Name
                "0010|0020", # Patient ID
                "0010|0030", # Patient Birth Date
                "0020|0010", # Study ID, for human consumption
                "0008|0020", # Study Date
                "0008|0030", # Study Time
                "0008|0050", # Accession Number
                "0008|0060"  # Modality
        ]

        direction = sitkImage.GetDirection()
        modification_time = time.strftime("%H%M%S")
        modification_date = time.strftime("%Y%m%d")        
        series_tag_values = [(k, self.tar_reader.GetMetaData(0,k)) for k in tags_to_copy if self.tar_reader.HasMetaDataKey(0,k)] + \
                        [("0008|0031",modification_time), # Series Time
                        ("0008|0021",modification_date), # Series Date
                        ("0008|0008","DERIVED\\SECONDARY"), # Image Type
                        ("0020|0011",str(new_series_n)), # SeriesNumber
                        ("0020|000e", self.tar_reader.GetMetaData(0, "0020|000e")+modification_date+".1"+modification_time), # Series Instance UID
                        ("0020|000d", self.tar_reader.GetMetaData(0, "0020|000e")+modification_date+".1"+modification_time), # Study Instance UID
                        ("0020|0037", '\\'.join(map(str, (direction[0], direction[3], direction[6],# Image Orientation (Patient)
                                                            direction[1],direction[4],direction[7])))),
                        ("0008|103e", self.tar_reader.GetMetaData(0,"0008|103e") + " Processed-SimpleITK"), # Series Description
                        ("0018|0088", self.ref_reader.GetMetaData(0, "0018|0088")), # SpacingBetweenSlices
                        ("0018|0050", self.ref_reader.GetMetaData(0, "0018|0050")) # SliceThickness
                        ] 
        
        for i, f in zip(range(sitkImage.GetDepth()), self.filenames):
            image_slice = sitkImage[:,:,i]
            # Tags shared by the series.
            for tag, value in series_tag_values:
                image_slice.SetMetaData(tag, value)
            # Slice specific tags.
            image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d")) # Instance Creation Date
            image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S")) # Instance Creation Time
            image_slice.SetMetaData("0020|0032", '\\'.join(map(str,sitkImage.TransformIndexToPhysicalPoint((0,0,i))))) # Image Position (Patient)
            image_slice.SetMetaData("0020,0013", str(i+1)) # Instance Number

            # Write to the output directory and add the extension dcm, to force writing in DICOM format.
            # writer.SetFileName(os.path.join(output_dir,"IM{:0>4d}.dcm".format(i+1) ))
            writer.SetFileName(os.path.join(output_dir,f))
            writer.Execute(image_slice)
        return

def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The directory path %s is not valid" % arg)
    else:
        return os.path.abspath(arg)  # return an abasolute path

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Coregistration')
    parser.add_argument('--ref', dest='ref_dir', required=True,
                    metavar="Directory", type=lambda x:is_valid_file(parser, x),
                    help='A valid path to the directory of reference image series. There should be only one dicom series')
    parser.add_argument('--tar', dest='tar_dir', required=True,
                    metavar="Directory", type=lambda x:is_valid_file(parser, x),
                    help='A valid path to the directory of target/moving image series. There should be only one dicom series')
    args = parser.parse_args()

    ref_dir = args.ref_dir
    tar_dir = args.tar_dir

    reg = Registration(ref_dir, tar_dir)
    reg.registration()
    reg.save_dicom(os.path.join(os.path.dirname(tar_dir), os.path.basename(tar_dir)+"_coregistered"))

    print("Reference Image Spacing:{}".format(reg.ref_img.GetSize()))
    print("Moving Image Spacing:{}".format(reg.tar_img.GetSize()))
    print("Co-registered Image Spacing:{}".format(reg.reg_img.GetSize()))