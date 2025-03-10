
t12Path = "meshes/meshes/T12_imfusion.vertebra.obj"
l1Path = "meshes/meshes/L1_imfusion.vertebra.obj"
l2Path = "meshes/meshes/L2_imfusion.vertebra.obj"
l3Path = "meshes/meshes/L3_imfusion.vertebra.obj"
l4Path = "meshes/meshes/L4_imfusion.vertebra.obj"
l5Path = "meshes/meshes/L5_imfusion.vertebra.obj"
s1Path = "meshes/meshes/Sacrum_imfusion.sacrum.obj"


def addHeader(rootNode):

    rootNode.addObject('VisualStyle', displayFlags='showVisualModels hideBehaviorModels hideForceFields hideWireframe')
    rootNode.addObject('RequiredPlugin', name='SofaPlugins', pluginName=['ArticulatedSystemPlugin', 'SofaPython3'])

    rootNode.addObject('DefaultVisualManagerLoop')
    rootNode.addObject('FreeMotionAnimationLoop')
    rootNode.addObject('GenericConstraintSolver', maxIterations=50, tolerance=1e-5, printLog=False)
    rootNode.addObject('BackgroundSetting', color=[1., 1., 1., 1.])
    rootNode.findData('dt').value=0.01
    rootNode.gravity = [0,-9810,0]


def addVisu(node, index, filename, translation=[0, 0, 0]):

    if filename is None:
        return

    visu = node.addChild('Visu'+str(index))
    visu.addObject('MeshOBJLoader', name='loader', filename=filename, translation=translation)
    visu.addObject('MeshTopology', src='@loader')
    visu.addObject('OglModel', color=[1.0,0.8,0.0,1.])
    visu.addObject('RigidMapping')

    return


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


def addPart(node, name, index, filename1, filename2=None, translation=[0, 0, 0]):

    part = node.addChild(name)
    part.addObject('MechanicalObject', template='Rigid3', position=[0,0,0,0,0,0,1])
    part.addObject('RigidMapping', index=index, globalToLocalCoords=True)

    addVisu(part, 1, filename1, translation=translation)
    addVisu(part, 2, filename2, translation=translation)

    return part


class Robot:

    def __init__(self, node):
        self.node=node

    def addRobot(self, name='Robot', translation=[0,0,0]):

        # Positions of parts
        positions = [
                    [160.8,     0, 160.8, 0,0,0,1],
                    [160.8,  78.5, 160.8, 0,0,0,1],
                    [254.8,   171, 160.8, 0,0,0,1],
                    [347.3,   372, 160.8, 0,0,0,1],
                    [254.8, 569.6, 160.8, 0,0,0,1],
                    [160.8, 500.5, 160.8, 0,0,0,1],
                    [1600.8, 442.5, 160.8, 0,0,0,1]
                    ]

        # You can change the joint angles here
        initAngles = [0,0,0,0,0,0]

        # Robot node
        robot = self.node.addChild(name)
        robot.addData('angles', initAngles, None, 'angle of articulations in radian', '', 'vector<float>')
        robot.addObject('EulerImplicitSolver')
        robot.addObject('SparseLDLSolver', template="CompressedRowSparseMatrixMat3x3d")
        robot.addObject('GenericConstraintCorrection')

        # Articulations node
        articulations = robot.addChild('Articulations')
        articulations.addObject('MechanicalObject', name='dofs', template='Vec1', rest_position=robot.getData('angles').getLinkPath(), position=initAngles)
        articulations.addObject('ArticulatedHierarchyContainer')
        articulations.addObject('UniformMass', totalMass=1)
        articulations.addObject('RestShapeSpringsForceField', stiffness=1e10, points=list(range(6)))

        # Rigid
        rigid = articulations.addChild('Rigid')
        rigid.addObject('MechanicalObject', name='dofs', template='Rigid3', showObject=False, showObjectScale=10,
                            position=positions[0:7],
                            translation=translation)
        rigid.addObject('ArticulatedSystemMapping', input1=articulations.dofs.getLinkPath(), output=rigid.dofs.getLinkPath())

        # Visu
        visu = rigid.addChild('Visu')
        addPart(visu, 'Base' , 0, t12Path, translation=translation)
        addPart(visu, 'Part1', 1, l1Path, translation=translation)
        addPart(visu, 'Part2', 2, l2Path, translation=translation)
        addPart(visu, 'Part3', 3, l3Path, translation=translation)
        addPart(visu, 'Part4', 4, l4Path, translation=translation)
        addPart(visu, 'Part5', 5, l5Path, translation=translation)
        addPart(visu, 'Part6', 6, s1Path, translation=translation)

        # Center of articulations
        centers = articulations.addChild('ArticulationsCenters')
        addCenter(centers, 'CenterBase' , 0, 1, [   0,  78.5, 0], [   0,      0, 0], 0, 0, 1, [0, 1, 0], 0)
        addCenter(centers, 'CenterPart1', 1, 2, [  94,  92.5, 0], [   0,      0, 0], 0, 0, 1, [1, 0, 0], 1)
        addCenter(centers, 'CenterPart2', 2, 3, [92.5,  92.5, 0], [   0, -108.5, 0], 0, 0, 1, [0, 1, 0], 2)
        addCenter(centers, 'CenterPart3', 3, 4, [   0, 108.5, 0], [92.5,  -89.1, 0], 0, 0, 1, [0, 0, 0], 3)
        addCenter(centers, 'CenterPart4', 4, 5, [   0,     0, 0], [  94,   69.1, 0], 0, 0, 1, [1, 0, 0], 4)
        addCenter(centers, 'CenterPart5', 5, 6, [   0,     0, 0], [   0,     58, 0], 0, 0, 1, [0, 1, 0], 5)

        return robot


# Test/example scene
def createScene(rootNode):
    
    # from robotGUI import RobotGUI  # Uncomment this if you want to use the GUI

    addHeader(rootNode)

    # Robot
    robot = Robot(rootNode).addRobot()
    # robot.addObject(RobotGUI(robot=robot))  # Uncomment this if you want to use the GUI

    return
