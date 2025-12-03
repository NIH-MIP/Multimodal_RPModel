# Contributor: Benjamin D. Simon and Katie Merriman
# Email: air@nih.gov
# Nov 26, 2024
#
# By downloading or otherwise receiving the SOFTWARE, RECIPIENT may 
# use and/or redistribute the SOFTWARE, with or without modification, 
# subject to RECIPIENT’s agreement to the following terms:
# 
# 1. THE SOFTWARE SHALL NOT BE USED IN THE TREATMENT OR DIAGNOSIS 
# OF CANINE OR HUMAN SUBJECTS.  RECIPIENT is responsible for 
# compliance with all laws and regulations applicable to the use 
# of the SOFTWARE.
# 
# 2. The SOFTWARE that is distributed pursuant to this Agreement 
# has been created by United States Government employees. In 
# accordance with Title 17 of the United States Code, section 105, 
# the SOFTWARE is not subject to copyright protection in the 
# United States.  Other than copyright, all rights, title and 
# interest in the SOFTWARE shall remain with the PROVIDER.   
# 
# 3.	RECIPIENT agrees to acknowledge PROVIDER’s contribution and 
# the name of the author of the SOFTWARE in all written publications 
# containing any data or information regarding or resulting from use 
# of the SOFTWARE. 
# 
# 4.	THE SOFTWARE IS PROVIDED "AS IS" AND ANY EXPRESS OR IMPLIED 
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES 
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT 
# ARE DISCLAIMED. IN NO EVENT SHALL THE PROVIDER OR THE INDIVIDUAL DEVELOPERS 
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF 
# THE POSSIBILITY OF SUCH DAMAGE.  
# 
# 5.	RECIPIENT agrees not to use any trademarks, service marks, trade names, 
# logos or product names of NCI or NIH to endorse or promote products derived 
# from the SOFTWARE without specific, prior and written permission.
# 
# 6.	For sake of clarity, and not by way of limitation, RECIPIENT may add its 
# own copyright statement to its modifications or derivative works of the SOFTWARE 
# and may provide additional or different license terms and conditions in its 
# sublicenses of modifications or derivative works of the SOFTWARE provided that 
# RECIPIENT’s use, reproduction, and distribution of the SOFTWARE otherwise complies 
# with the conditions stated in this Agreement. Whenever Recipient distributes or 
# redistributes the SOFTWARE, a copy of this Agreement must be included with 
# each copy of the SOFTWARE.

import SimpleITK as sitk
import numpy as np
import csv
import os
from scipy.ndimage import label


class EPEdetector:
    def __init__(self):

        self.threshold1 = 0.35  # threshold for lesion mask is 0.6344772701607316 , previously tried .35
        self.threshold2 = 0.45
        self.threshold3 = 0.55
        self.threshold4 = 0.65
        self.testNum = 'test'

        ### Set Path to Folder Containing Monai Output Masks
        self.mask_folder = "path/to/monai_output"

        ### Set Path to 
        self.fileName1 = "lesion_data.csv"

    def getEPEdata(self):

        file = open(self.fileName1, 'a+', newline='')
        # writing the data into the file
        with file:
            write = csv.writer(file)
            write.writerows([['lesion', 'insideCapsuleArea', 'outsideCapsuleArea', 'outsideVarianceArea',
                      'distfromCapsule', 'distfromCapsule_xy', 'distfromCapsule_z', 'prostCoords',
                      'distfromVar', 'distfromVar_xy', 'distfromVar_z', 'varCoords',
                      'distInsideCapsule', 'distInsideCapsule_xy', 'distInsideCapsule_z', 'insideCoords']])

        file.close()

        ### Set Accoriing to naming convention SURG-00X with range accordingly
        for p in reversed(range(1, 635)):

            # Patient name should follow format 'SURG-00X'
            patient = 'SURG-' + str(p + 1000)[1:]
            print(patient)

            try:
                prostPath = os.path.join(self.mask_folder, patient, 'wp_bt_undilated.nii.gz')
                prostImg = sitk.ReadImage(prostPath)
                prostArr = sitk.GetArrayFromImage(prostImg)

                spacing = prostImg.GetSpacing()

                file = open(self.fileName1, 'a+', newline='')
                # writing the data into the file
                with file:
                    write = csv.writer(file)
                    write.writerows([[patient, spacing]])
                file.close()

                prostEdgeImg = sitk.ReadImage(os.path.join(self.mask_folder, patient, "wp_prostEdge.nii.gz"))
                prostEdgeArr = sitk.GetArrayFromImage(prostEdgeImg)
                varImg = sitk.ReadImage(os.path.join(self.mask_folder, patient, "wp_fullVar.nii.gz"))
                varArr = sitk.GetArrayFromImage(varImg)
                varEdgeImg = sitk.ReadImage(os.path.join(self.mask_folder, patient, "wp_outsideVarEdge.nii.gz"))
                varEdgeArr = sitk.GetArrayFromImage(varEdgeImg)
                insideImg = sitk.ReadImage(os.path.join(self.mask_folder, patient, "wp_bt_inside.nii.gz"))
                insideArr = sitk.GetArrayFromImage(insideImg)
                insideEdgeImg = sitk.ReadImage(os.path.join(self.mask_folder, patient, "wp_insideVarEdge.nii.gz"))
                insideEdgeArr = sitk.GetArrayFromImage(insideEdgeImg)
                varZoneImg = sitk.ReadImage(os.path.join(self.mask_folder, patient, "wp_bt_fullVarZone.nii.gz"))
                varZoneArr = sitk.GetArrayFromImage(varZoneImg)

                self.createBinaryLesions(patient)
                print("binaries created")

                lesionMask = sitk.ReadImage(os.path.join(self.mask_folder, patient, 'lesion', 'lesion_mask.nii'))
                lesionArr = sitk.GetArrayFromImage(lesionMask)


                print("beginning data calculation")
                self.lesionData(patient, prostArr, prostEdgeArr, varArr, varEdgeArr,
                                insideArr, insideEdgeArr, lesionArr, varZoneArr, "1", "orig")
                self.lesionData(patient, prostArr, prostEdgeArr, varArr, varEdgeArr,
                                insideArr, insideEdgeArr, lesionArr, varZoneArr, "2", "orig")
                self.lesionData(patient, prostArr, prostEdgeArr, varArr, varEdgeArr,
                                insideArr, insideEdgeArr, lesionArr, varZoneArr, "3", "orig")
                self.lesionData(patient, prostArr, prostEdgeArr, varArr, varEdgeArr,
                                insideArr, insideEdgeArr, lesionArr, varZoneArr, "4", "orig")

            except RuntimeError:
                print("remote error")
                file = open(self.fileName1, 'a+', newline='')
                # writing the data into the file
                with file:
                    write = csv.writer(file)
                    write.writerows([[patient, "remote error"]])
                file.close()
        return

    def createEdge(self, patient, prost, prostArr, suffix):

        arr_shape = prostArr.shape
        prostNZ = prostArr.nonzero()  # saved as tuple in z,y,x order
        capsule = np.zeros(arr_shape, dtype=int)

        # Find array of x,y,z tuples corresponding to voxels of prostNZ that are on edge of prostate array
        # and also adjacent to lesion voxels outside of prostate
        for prostVoxel in range(len(prostNZ[0])):
            # If voxel above or below current voxel is 0, voxel is on the edge
            # if that voxel contains lesion, voxel is portion of capsule with lesion contact
            if (prostNZ[0][prostVoxel] - 1) > -1:
                if prostArr[prostNZ[0][prostVoxel] - 1, prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]] == 0:
                    capsule[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]] = 1
            if (prostNZ[0][prostVoxel]) < (arr_shape[0] - 1):
                if prostArr[prostNZ[0][prostVoxel] + 1, prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]] == 0:
                    capsule[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]] = 1
            # if voxel anterior or posterior of current voxel is 0, voxel is on the edge
            if (prostNZ[1][prostVoxel] - 1) > -1:
                if prostArr[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel] - 1, prostNZ[2][prostVoxel]] == 0:
                    capsule[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]] = 1
            if (prostNZ[1][prostVoxel]) < (arr_shape[1] - 1):
                if prostArr[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel] + 1, prostNZ[2][prostVoxel]] == 0:
                    capsule[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]] = 1
            # if voxel to right or left of current voxel is 0, voxel is on the edge
            if (prostNZ[2][prostVoxel] - 1) > -1:
                if prostArr[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel], prostNZ[2][prostVoxel] - 1] == 0:
                    capsule[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]] = 1
            if (prostNZ[2][prostVoxel]) < (arr_shape[2] - 1):
                if prostArr[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel], prostNZ[2][prostVoxel] + 1] == 0:
                    capsule[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]] = 1

        # save edge mask to folder
        EPEmaskfolder = os.path.join(self.mask_folder, patient, self.testNum)
        if not os.path.exists(EPEmaskfolder):
            os.mkdir(EPEmaskfolder)
        newname = os.path.join(EPEmaskfolder, patient + suffix)
        edgeMask = sitk.GetImageFromArray(capsule)
        edgeMask.CopyInformation(prost)
        sitk.WriteImage(edgeMask, newname)

        return capsule


    def createBinaryLesions(self, patient):

        ### Create Binary Lesions at each threshold
        saveFolder = os.path.join(self.mask_folder, patient, self.testNum)
        if not os.path.exists(saveFolder):
            os.mkdir(saveFolder)

        lesionHeatMap = sitk.ReadImage(os.path.join(self.mask_folder, patient, 'lesion', 'merged_lesion_prob.nii'))
        probArr = sitk.GetArrayFromImage(lesionHeatMap)

        # save binary lesion mask for threshold 1
        binaryArr = np.where(probArr > self.threshold1, 1, 0)
        binaryname = os.path.join(saveFolder, patient + '_allLesions_thresh1.nii.gz')
        binarymask = sitk.GetImageFromArray(binaryArr)
        binarymask.CopyInformation(lesionHeatMap)
        sitk.WriteImage(binarymask, binaryname)

        # save binary lesion mask for threshold 2
        binaryArr = np.where(probArr > self.threshold2, 1, 0)
        binaryname = os.path.join(saveFolder, patient + '_allLesions_thresh2.nii.gz')
        binarymask = sitk.GetImageFromArray(binaryArr)
        binarymask.CopyInformation(lesionHeatMap)
        sitk.WriteImage(binarymask, binaryname)

        # save binary lesion mask for threshold 3
        binaryArr = np.where(probArr > self.threshold3, 1, 0)
        binaryname = os.path.join(saveFolder, patient + '_allLesions_thresh3.nii.gz')
        binarymask = sitk.GetImageFromArray(binaryArr)
        binarymask.CopyInformation(lesionHeatMap)
        sitk.WriteImage(binarymask, binaryname)

        # save binary lesion mask for threshold 4
        binaryArr = np.where(probArr > self.threshold4, 1, 0)
        binaryname = os.path.join(saveFolder, patient + '_allLesions_thresh4.nii.gz')
        binarymask = sitk.GetImageFromArray(binaryArr)
        binarymask.CopyInformation(lesionHeatMap)
        sitk.WriteImage(binarymask, binaryname)

        return

    def lesionData(self, patient, prostArr, prostEdge, varArr, varEdge, insideArr, insideEdge, lesionArr,
                   varianceZoneArr, num, version):
        
        ### Aggregate Lesion Masks with respect to variance zone to create final measurements

        saveFolder = os.path.join(self.mask_folder, patient, self.testNum)
        allLesions = sitk.ReadImage(os.path.join(saveFolder, patient + '_allLesions_thresh' + num + '.nii.gz'))
        binaryArr = sitk.GetArrayFromImage(allLesions)
        # create labeled array separating individual lesions
        labeled_array, num_features = label(binaryArr)
        labeledname = os.path.join(saveFolder, patient + '_lesions_labeled_thresh' + num + '.nii.gz')
        labeledmask = sitk.GetImageFromArray(labeled_array)
        labeledmask.CopyInformation(allLesions)
        sitk.WriteImage(labeledmask, labeledname)

        spacing = allLesions.GetSpacing()

        patientData = []

        file = open(os.path.join(os.path.dirname(self.mask_folder), 'threshold' + num + version + 'remote.csv'), 'a+',
                    newline='')
        with file:
            write = csv.writer(file)
            write.writerows([[]])
            write.writerows([[patient]])
        file.close()

        for i in range(num_features):
            val = i + 1
            print(patient + " threshold: " + num + version + " lesion: " + str(val))

            distfromCapsule = -1
            distfromCapsule_xy = -1
            distfromCapsule_z = -1
            distfromVar = -1
            distfromVar_xy = -1
            distfromVar_z = -1
            distInsideCapsule = -1
            distInsideCapsule_xy = -1
            distInsideCapsule_z = -1
            outsideCapsule = 0
            outsideVarArea = 0
            insideCapsuleArea = 0
            probLesionArr = np.where(labeled_array == val, 1, 0)
            probLesionname = os.path.join(saveFolder, patient + '_lesion' + str(val) + '_thresh' + num + '.nii.gz')
            probLesionmask = sitk.GetImageFromArray(probLesionArr)
            probLesionmask.CopyInformation(allLesions)
            sitk.WriteImage(probLesionmask, probLesionname)

            # check that lesionMask overlaps with probLesion array
            lesionNZ = probLesionArr.nonzero()
            excludeLesion = 1
            for ind in range(len(lesionNZ[0])):
                if lesionArr[lesionNZ[0][ind], lesionNZ[1][ind], lesionNZ[2][ind]] == 1:
                    excludeLesion = 0
                    break

            # if lesion does match lesion on mask:
            if excludeLesion == 0:
                outsideVarArr = []
                outsideProstArr = []
                insideProstArr = []
                prostCoordsTemp = []
                prostCoords = []
                varCoordsTemp = []
                varCoords = []
                insideCoordsTemp = []
                insideCoords = []

                # create array of lesion outside of outerVariance zone, prostate, and innerVariance zone respectively,
                # then create array of edges for those arrays
                outsideVarArr = np.where(varArr == 0, probLesionArr, 0)
                outsideVarEdge = self.createEdge(patient, allLesions, outsideVarArr,
                                                 "_outsideVarEdge_" + version + '_thresh' + num + "_lesion" + str(
                                                     val) + ".nii.gz")
                outsideVarNZ = outsideVarEdge.nonzero()

                outsideProstArr = np.where(prostArr == 0, probLesionArr, 0)
                outsideProstEdge = self.createEdge(patient, allLesions, outsideProstArr,
                                                   "_outsideProstEdge_" + version + '_thresh' + num + "_lesion" + str(
                                                       val) + ".nii.gz")
                outsideProstNZ = outsideProstEdge.nonzero()

                # this array is full portion of lesion that's inside prostate
                insideProstArr = np.where(prostArr == 1, probLesionArr, 0)
                #this array is portion of lesion that's inside prostate but within variance zone
                innerVarArr = np.where(insideArr == 0, probLesionArr, 0)
                insideProstEdge = self.createEdge(patient, allLesions, innerVarArr,
                                                  "_insideProstEdge_" + version + '_thresh' + num + "_lesion" + str(
                                                      val) + ".nii.gz")
                innerVarNZ = insideProstEdge.nonzero()


                outsideCapsuleArea = len(outsideProstArr.nonzero()[0])
                print("outsideCapsule new: ", outsideCapsuleArea)
                outsideVarianceArea = len(outsideVarArr.nonzero()[0])
                print("outsideVariance new: ", outsideVarianceArea)
                insideCapsuleArea = len(insideProstArr.nonzero()[0])
                print("insideCapsule new: ", insideCapsuleArea)


                # should be unneccessary given earlier check against lesion map.  May want to exclude in future versions
                if (outsideCapsuleArea > 2*insideCapsuleArea) and (version == "orig"):
                    # if more than 2/3 of the lesion is outside of the organ (likely false call):
                    print("more outside than inside")
                    continue

                ## Create portion of capsule/outerVariance/innerVariance
                # that lesion passes through to check distance against:
                if outsideCapsuleArea != 0:
                    prostIntersectArr = np.where(probLesionArr == 1, prostEdge, 0)
                    prostIntersectImg = sitk.GetImageFromArray(prostIntersectArr)
                    prostIntersectImg.CopyInformation(allLesions)
                    sitk.WriteImage(prostIntersectImg, os.path.join(saveFolder, "capsuleIntersection_thresh" + version
                                                                + "_lesion" + num + "_" + str(val) + ".nii.gz"))
                    prostEdgeNZ = prostIntersectArr.nonzero()

                    if outsideVarianceArea != 0:
                        varIntersectArr = np.where(probLesionArr == 1, varEdge, 0)
                        varIntersectImg = sitk.GetImageFromArray(varIntersectArr)
                        varIntersectImg.CopyInformation(allLesions)
                        sitk.WriteImage(varIntersectImg, os.path.join(saveFolder,
                                                     "varianceIntersection_thresh" + version
                                                     + "_lesion" + num + "_" + str(val) + ".nii.gz"))
                        varEdgeNZ = varIntersectArr.nonzero()
                else:
                    prostIntersectArr = np.where(probLesionArr == 1, prostEdge, 0)
                    prostEdgeNZ = prostIntersectArr.nonzero()
                    insideIntersectArr = np.where(probLesionArr == 1, insideEdge, 0)
                    insideIntersectImg = sitk.GetImageFromArray(insideIntersectArr)
                    insideIntersectImg.CopyInformation(allLesions)
                    sitk.WriteImage(insideIntersectImg, os.path.join(saveFolder, "insideIntersection_thresh" + version
                                                                + "_lesion" + num + "_" + str(val) + ".nii.gz"))
                    insideEdgeNZ = insideIntersectArr.nonzero()
                    if len(insideEdgeNZ[0]) == 0: # if the lesion doesn't intersect with the inner variance edge
                        insideEdgeNZ = insideEdge.nonzero()



                ## Find distance away from capsule
                if outsideCapsuleArea != 0:
                    print('len outsideProstEdge:', len(outsideProstNZ[0]))
                    # if lesion outside of prostate variance:
                    for vox in range(len(outsideProstNZ[0])):
                        min_dist = 256
                        min_dist_xy = 256
                        min_dist_z = 256

                        if vox % 50 == 0:
                            print('vox', vox)
                        for prostVox in range(len(prostEdgeNZ[0])):
                            dist = np.sqrt((spacing[2] * (outsideProstNZ[0][vox] - prostEdgeNZ[0][prostVox])) ** 2 +
                                           (spacing[1] * (outsideProstNZ[1][vox] - prostEdgeNZ[1][prostVox])) ** 2 +
                                           (spacing[0] * (outsideProstNZ[2][vox] - prostEdgeNZ[2][prostVox])) ** 2)
                            dist_xy = np.sqrt((spacing[1] * (outsideProstNZ[1][vox] - prostEdgeNZ[1][prostVox])) ** 2 +
                                              (spacing[0] * (outsideProstNZ[2][vox] - prostEdgeNZ[2][prostVox])) ** 2)
                            dist_z = spacing[2] * abs(outsideProstNZ[0][vox] - prostEdgeNZ[0][prostVox])
                            # save full coordinate info for point with minimum 3D distance
                            if dist < min_dist:
                                min_dist = dist
                                min_dist_xy = dist_xy
                                min_dist_z = dist_z
                                prostCoordsTemp = [
                                    'lesion:' + str(outsideProstNZ[0][vox]) + ',' + str(outsideProstNZ[1][vox]) + ',' +
                                    str(outsideProstNZ[2][vox]) + ', prostate: ' + str(prostEdgeNZ[0][prostVox])
                                    + ',' + str(prostEdgeNZ[1][prostVox]) + ',' + str(prostEdgeNZ[2][prostVox])]

                        if min_dist > distfromCapsule:
                            distfromCapsule = min_dist
                            distfromCapsule_xy = min_dist_xy
                            distfromCapsule_z = min_dist_z
                            prostCoords = prostCoordsTemp


                    if outsideVarianceArea != 0:
                        print('len outsideVarEdge:', len(outsideVarNZ[0]))

                        # if lesion outside of prostate variance:
                        for vox in range(len(outsideVarNZ[0])):
                            min_dist = 256
                            min_dist_xy = 256
                            min_dist_z = 256

                            if vox % 50 == 0:
                                print('vox', vox)
                            for prostVox in range(len(varEdgeNZ[0])):
                                dist = np.sqrt((spacing[2] * (outsideVarNZ[0][vox] - varEdgeNZ[0][prostVox])) ** 2 +
                                               (spacing[1] * (outsideVarNZ[1][vox] - varEdgeNZ[1][prostVox])) ** 2 +
                                               (spacing[0] * (outsideVarNZ[2][vox] - varEdgeNZ[2][prostVox])) ** 2)
                                dist_xy = np.sqrt(
                                    (spacing[1] * (outsideVarNZ[1][vox] - varEdgeNZ[1][prostVox])) ** 2 +
                                    (spacing[0] * (outsideVarNZ[2][vox] - varEdgeNZ[2][prostVox])) ** 2)
                                dist_z = spacing[2] * abs(outsideVarNZ[0][vox] - varEdgeNZ[0][prostVox])
                                # save full coordinate info for point with minimum 3D distance
                                if dist < min_dist:
                                    min_dist = dist
                                    min_dist_xy = dist_xy
                                    min_dist_z = dist_z
                                    varCoordsTemp = [
                                        'lesion:' + str(outsideVarNZ[0][vox]) + ',' + str(outsideVarNZ[1][vox]) + ',' +
                                        str(outsideVarNZ[2][vox]) + ', prostate: ' + str(varEdgeNZ[0][prostVox])
                                        + ',' + str(varEdgeNZ[1][prostVox]) + ',' + str(varEdgeNZ[2][prostVox])]

                            if min_dist > distfromVar:
                                distfromVar = min_dist
                                distfromVar_xy = min_dist_xy
                                distfromVar_z = min_dist_z
                                varCoords = varCoordsTemp


                else:
                    print('len insideProstEdge:', len(innerVarNZ[0]))
                    if len(prostEdgeNZ[0]) != 0: # if lesion intersects with but doesn't go past capsule edge:
                        distfromCapsule = 0
                        distfromCapsule_xy = 0
                        distfromCapsule_z = 0
                        prostCoords = ['intersection:' + str(prostEdgeNZ[0][0]) + ',' +
                                       str(prostEdgeNZ[1][0]) + ',' + str(prostEdgeNZ[2][0])]

                        distfromVar = 0
                        distfromVar_xy = 0
                        distfromVar_z = 0
                        varCoords = ['intersection:' + str(prostEdgeNZ[0][0]) + ',' +
                                       str(prostEdgeNZ[1][0]) + ',' + str(prostEdgeNZ[2][0])]

                        distInsideCapsule = 10
                        distInsideCapsule_xy = 10
                        distInsideCapsule_z = 10
                        insideCoords = ['intersection:' + str(prostEdgeNZ[0][0]) + ',' +
                                       str(prostEdgeNZ[1][0]) + ',' + str(prostEdgeNZ[2][0])]


                    elif len(innerVarNZ[0]) == 0:
                        distfromCapsule = 0
                        distfromCapsule_xy = 0
                        distfromCapsule_z = 0
                        prostCoords = ['organ confined']

                        distfromVar = 0
                        distfromVar_xy = 0
                        distfromVar_z = 0
                        varCoords = ['organ confined']

                        distInsideCapsule = 0
                        distInsideCapsule_xy = 0
                        distInsideCapsule_z = 0
                        insideCoords = ['organ confined']

                    else:
                        for vox in range(len(innerVarNZ[0])):
                            min_dist = 256
                            min_dist_xy = 256
                            min_dist_z = 256

                            if vox % 50 == 0:
                                print('vox', vox)
                            for prostVox in range(len(insideEdgeNZ[0])):
                                dist = np.sqrt((spacing[2] * (innerVarNZ[0][vox] - insideEdgeNZ[0][prostVox])) ** 2 +
                                               (spacing[1] * (innerVarNZ[1][vox] - insideEdgeNZ[1][prostVox])) ** 2 +
                                               (spacing[0] * (innerVarNZ[2][vox] - insideEdgeNZ[2][prostVox])) ** 2)
                                dist_xy = np.sqrt((spacing[1] * (innerVarNZ[1][vox] - insideEdgeNZ[1][prostVox])) ** 2 +
                                                  (spacing[0] * (innerVarNZ[2][vox] - insideEdgeNZ[2][prostVox])) ** 2)
                                dist_z = spacing[2] * abs(innerVarNZ[0][vox] - insideEdgeNZ[0][prostVox])
                                # save full coordinate info for point with minimum 3D distance
                                if dist < min_dist:
                                    min_dist = dist
                                    min_dist_xy = dist_xy
                                    min_dist_z = dist_z
                                    insideCoordsTemp = [
                                        'lesion:' + str(innerVarNZ[0][vox]) + ',' + str(innerVarNZ[1][vox]) + ',' +
                                        str(innerVarNZ[2][vox]) + ', prostate: ' + str(insideEdgeNZ[0][prostVox])
                                        + ',' + str(insideEdgeNZ[1][prostVox]) + ',' + str(insideEdgeNZ[2][prostVox])]

                            if min_dist > distInsideCapsule:
                                distInsideCapsule = min_dist
                                distInsideCapsule_xy = min_dist_xy
                                distInsideCapsule_z = min_dist_z
                                insideCoords = insideCoordsTemp


                if distInsideCapsule>-1:
                    distInsideCapsule = distInsideCapsule-10
                    distInsideCapsule_xy = distInsideCapsule_xy-10
                    distInsideCapsule_z = distInsideCapsule_z-10
                else:
                    distInsideCapsule = "n/a"
                    distInsideCapsule_xy = "n/a"
                    distInsideCapsule_z = "n/a"


                lesionData = ['lesion' + str(val), insideCapsuleArea, outsideCapsuleArea, outsideVarianceArea,
                      distfromCapsule, distfromCapsule_xy, distfromCapsule_z, prostCoords,
                      distfromVar, distfromVar_xy, distfromVar_z, varCoords,
                      distInsideCapsule, distInsideCapsule_xy, distInsideCapsule_z, insideCoords]

                print(lesionData)

                file = open(os.path.join(os.path.dirname(self.mask_folder), 'threshold' + num + version + 'remote.csv'), 'a+',
                            newline='')
                with file:
                    write = csv.writer(file)
                    write.writerows([[patient, 'lesion' + str(val), insideCapsuleArea, outsideCapsuleArea, outsideVarianceArea,
                      distfromCapsule, distfromCapsule_xy, distfromCapsule_z, prostCoords,
                      distfromVar, distfromVar_xy, distfromVar_z, varCoords,
                      distInsideCapsule, distInsideCapsule_xy, distInsideCapsule_z, insideCoords]])
                file.close()
        return


if __name__ == '__main__':
    c = EPEdetector()
    c.getEPEdata()
    #    c.create_csv_files()
    print('Check successful')
