# Required import for python
import Sofa
import Sofa.Core
import json


def xyzToArray(xyz):
    
    return [xyz['x'], xyz['y'], xyz['z']]


def revert_pos(xyz):
    
    return [-1 * i for i in xyz]


def addHeader(rootNode):

    rootNode.addObject('VisualStyle', displayFlags='showVisualModels hideBehaviorModels hideForceFields')
    rootNode.addObject('RequiredPlugin', name='SofaPlugins', pluginName=['ArticulatedSystemPlugin', 'SofaPython3'])

    # rootNode.addObject('DefaultVisualManagerLoop')
    # rootNode.addObject('FreeMotionAnimationLoop')
    # rootNode.addObject('GenericConstraintSolver', maxIterations=50, tolerance=1e-5, printLog=False)
    rootNode.addObject('DefaultAnimationLoop')
    rootNode.addObject('EulerImplicitSolver', rayleighStiffness=0.1, rayleighMass=0.1)
    rootNode.addObject('CGLinearSolver', iterations=100, tolerance=1e-7, threshold=1e-7)
    rootNode.addObject('BackgroundSetting', color=[0., 0., 0., 1.])
    rootNode.findData('dt').value=0.01
    rootNode.gravity = [0, 0,0]
        

def addVisu(node, index, filename, translation=[0, 0, 0]):

    if filename is None:
        return

    visu = node.addChild('Visu'+str(index))
    visu.addObject('MeshOBJLoader', name='loader', triangulate="true", filename=filename, translation=translation)
    visu.addObject('MeshTopology', src='@loader')
    visu.addObject('OglModel', src = '@loader', color=[1.0,0.8,0.0,1.])
    visu.addObject('RigidMapping')

    return visu

def addPart(node, name, index, filename1, filename2=None, translation=[0, 0, 0]):

    part = node.addChild(name)
    part.addObject('MechanicalObject', template='Rigid3', position=[0,0,0,0,0,0,1])
    # part.addObject('AngularSpringForceField', angularStiffness='1e4', name='ASF')
    part.addObject('RigidMapping', index=index, globalToLocalCoords=True)

    addVisu(part, 1, filename1, translation=translation)
    
    if filename2 is not None:
        addVisu(part, 2, filename2, translation=translation)

    return part


def addCenter(node, name,
              parentIndex, childIndex,
              posOnParent, posOnChild,
              articulationProcess,
              isTranslation, isRotation, axis,
              articulationIndex):

    center = node.addChild(name)
    center.addObject('ArticulationCenter', parentIndex=parentIndex, childIndex=childIndex, posOnParent=posOnParent, posOnChild=posOnChild, articulationProcess=articulationProcess)

    articulation = center.addChild('Articulation')
    articulation.addObject('Articulation', translation=isTranslation, rotation=isRotation, rotationAxis=axis, articulationIndex=articulationIndex)

    return center


def addAngularSpring(node):

    return


class LSpine:

    def __init__(self, node):
        self.node=node

    def addLSpine(self, name='LSpine', translation=[0,0,0]):

        metadata = json.load(open(r'C:\Users\wangs\dev\python_code\SOFA\meshes\kpts_data.json', 'r'))
        # Positions of parts
        positions = {i:xyzToArray(metadata['vertebrae_dissection'][i]['SupEndplateCenter']) 
                     for i in ['Sacrum', 'L5', 'L4', 'L3', 'L2', 'L1', 'T12', 'T11', 'T10', 'T9']
                     }

        # You can change the joint angles here
        initAngles = [0] * 9

        # spine node
        spine = self.node.addChild(name)
        spine.addData('angles', initAngles, None, 'angle of articulations in radian', '', 'vector<float>')
        spine.addObject('EulerImplicitSolver')
        spine.addObject('SparseLDLSolver', template="CompressedRowSparseMatrixMat3x3d")
        spine.addObject('GenericConstraintCorrection')

        # Articulations node
        articulations = spine.addChild('Articulations')
        articulations.addObject('MechanicalObject', name='dofs', template='Vec1', rest_position=spine.getData('angles').getLinkPath(), position=initAngles)
        articulations.addObject('ArticulatedHierarchyContainer')
        articulations.addObject('UniformMass', totalMass=1)
        # articulations.addObject('RestShapeSpringsForceField', stiffness=1e3, angularStiffness=1e3, points=list(range(8)))
        # articulations.addObject('AngularSpringForceField', angularStiffness=1e4, indices='0 1 2 3 4 5', name='ASF')

        # Rigid
        rigid = articulations.addChild('Rigid')
        rigid.addObject('MechanicalObject', name='dofs', template='Rigid3', showObject=False, showObjectScale=10,
                            position=[revert_pos(positions[i]) + [0, 0, 0, 1] for i in ['Sacrum', 'L5', 'L4', 'L3', 'L2', 'L1', 'T12', 'T11', 'T10', 'T9']],
                            translation=translation)
        rigid.addObject('ArticulatedSystemMapping', input1=articulations.dofs.getLinkPath(), output=rigid.dofs.getLinkPath())
        rigid.addObject('AngularSpringForceField', angularStiffness=1e4, indices='0 1 2 3 4 5 6 7 8 9', name='ASF')
        # rigid.addObject('RestShapeSpringsForceField', stiffness=10, angularStiffness=10, points=list(range(10)), mstate='@dofs', template='Rigid3')

        # collision = rigid.addChild('Collision')
        # collision.addObject('CubeTopology', max=[9.2,0.05,0.05], min='0,-0.05,-0.05',nx=70, ny=2, nz=2)
        # collision.addObject('MechanicalObject', name='skinDOFs')
        # collision.addObject('TriangleCollisionModel')
        # collision.addObject('LineCollisionModel')
        # collision.addObject('PointCollisionModel')
        # collision.addObject('BeamLinearMapping', isMechanical='true')


        # Visu
        visu = rigid.addChild('Visu')
        addPart(visu, 'T9' , 9, "meshes/meshes/T9_imfusion.vertebra.obj", translation=revert_pos(positions['T9']))
        addPart(visu, 'T10' , 8, "meshes/meshes/T10_imfusion.vertebra.obj", translation=revert_pos(positions['T10']))
        addPart(visu, 'T11' , 7, "meshes/meshes/T11_imfusion.vertebra.obj", translation=revert_pos(positions['T11']))
        addPart(visu, 'T12' , 6, "meshes/meshes/T12_imfusion.vertebra.obj", translation=revert_pos(positions['T12']))
        addPart(visu, 'L1' , 5, "meshes/meshes/L1_imfusion.vertebra.obj", translation=revert_pos(positions['L1']))
        addPart(visu, 'L2', 4, "meshes/meshes/L2_imfusion.vertebra.obj", translation=revert_pos(positions['L2']))
        addPart(visu, 'L3', 3, "meshes/meshes/L3_imfusion.vertebra.obj", translation=revert_pos(positions['L3']))
        addPart(visu, 'L4', 2, "meshes/meshes/L4_imfusion.vertebra.obj", translation=revert_pos(positions['L4']))
        addPart(visu, 'L5', 1, "meshes/meshes/L5_imfusion.vertebra.obj", translation=revert_pos(positions['L5']))
        addPart(visu, 'S1', 0, "meshes/meshes/Sacrum_imfusion.sacrum.obj", translation=revert_pos(positions['Sacrum']))
        # addPart(visu, 'Part5', 5, part51Path, part52Path, translation=translation)
        # addPart(visu, 'Part6', 6, part6Path, translation=translation)

        # Center of articulations
        centers = articulations.addChild('ArticulationsCenters')
        addCenter(centers, 'T9/T10', 8, 9, 
                  positions['T10'], 
                  positions['T10'], 
                  0, 0, 1, [1, 0, 0], 8)
        addCenter(centers, 'T10/T11', 7, 8, 
                  positions['T11'], 
                  positions['T11'], 
                  0, 0, 1, [1, 0, 0], 7)
        addCenter(centers, 'T11/T12', 6, 7, 
                  positions['T12'], 
                  positions['T12'], 
                  0, 0, 1, [1, 0, 0], 6)
        addCenter(centers, 'T12/L1', 5, 6, 
                  positions['L1'], 
                  positions['L1'], 
                  0, 0, 1, [1, 0, 0], 5)
        addCenter(centers, 'L1/L2', 4, 5, 
                  positions['L2'], 
                  positions['L2'], 
                  0, 0, 1, [1, 0, 0], 4)
        addCenter(centers, 'L2/L3', 3, 4, 
                  positions['L3'], 
                  positions['L3'], 
                  0, 0, 1, [1, 0, 0], 3)
        addCenter(centers, 'L3/L4', 2, 3, 
                  positions['L4'], 
                  positions['L4'], 
                  0, 0, 1, [1, 0, 0], 2)
        addCenter(centers, 'L4/L5', 1, 2, 
                  positions['L5'], 
                  positions['L5'], 
                  0, 0, 1, [1, 0, 0], 1)
        addCenter(centers, 'L5/S1', 0, 1, 
                  positions['Sacrum'], 
                  positions['Sacrum'], 
                  0, 0, 1, [1, 0, 0], 0)

        return spine
        
# Test/example scene
def createScene(rootNode):

    from controllerGUI import RobotGUI  # Uncomment this if you want to use the GUI

    addHeader(rootNode)

    # Robot
    robot = LSpine(rootNode).addLSpine()
    robot.addObject(RobotGUI(robot=robot))  # Uncomment this if you want to use the GUI

    return