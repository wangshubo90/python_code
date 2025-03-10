import numpy as np
import trimesh
from vtk import vtkPolyData, vtkPoints, vtkCellArray, vtkPolyDataWriter
from plyfile import PlyData, PlyElement
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def generate_rod_mesh(screw_centers: np.ndarray, diameter:float, output_name: str):
    # Parameterize the screw head positions
    t = np.linspace(0, 1, len(screw_centers))

    # Create cubic splines for each coordinate
    cs_x = CubicSpline(t, screw_centers[:, 0], bc_type='natural')
    cs_y = CubicSpline(t, screw_centers[:, 1], bc_type='natural')
    cs_z = CubicSpline(t, screw_centers[:, 2], bc_type='natural')

    # Generate smooth curve points
    t_fine = np.linspace(0, 1, 100)
    x_fine = cs_x(t_fine)
    y_fine = cs_y(t_fine)
    z_fine = cs_z(t_fine)

    # Plot the resulting curve
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_fine, y_fine, z_fine, label='Smooth Rod Curve')
    ax.scatter(screw_centers[:, 0], screw_centers[:, 1], screw_centers[:, 2], color='red', label='Screw Heads')
    ax.legend()
    plt.show()


    # =================================================


    # Calculate tangent, normal, and binormal vectors
    T = np.array([np.gradient(x_fine, t_fine), np.gradient(y_fine, t_fine), np.gradient(z_fine, t_fine)]).T
    T = T / np.linalg.norm(T, axis=1, keepdims=True)  # Normalize tangent vector
    B = np.cross(T, np.array([0, 0, 1]))  # Arbitrary binormal
    B = B / np.linalg.norm(B, axis=1, keepdims=True)
    N = np.cross(B, T)  # Normal vector

    # Create the tubular surface mesh
    vertices = []
    faces = []
    # Number of points around the tube for mesh generation
    N_points = 20
    d = diameter  # Tube diameter

    # Generate vertices around the tube
    for i in range(len(t_fine)):
        circle = []
        for j in range(N_points):
            angle = 2 * np.pi * j / N_points
            circle.append([
                x_fine[i] + (d / 2) * (N[i, 0] * np.cos(angle) + B[i, 0] * np.sin(angle)),
                y_fine[i] + (d / 2) * (N[i, 1] * np.cos(angle) + B[i, 1] * np.sin(angle)),
                z_fine[i] + (d / 2) * (N[i, 2] * np.cos(angle) + B[i, 2] * np.sin(angle))
            ])
        vertices.append(circle)
        if i > 0:
            for j in range(N_points):
                faces.append([i * N_points + j, i * N_points + (j + 1) % N_points, (i - 1) * N_points + j])
                faces.append([i * N_points + (j + 1) % N_points, (i - 1) * N_points + (j + 1) % N_points, (i - 1) * N_points + j])

    # Convert to NumPy array
    vertices = np.array(vertices).reshape(-1, 3)

    # Add caps to the ends
    # First end cap
    end1_center = np.array([x_fine[0], y_fine[0], z_fine[0]])
    vertices = np.vstack([vertices, end1_center])
    end1_center_index = len(vertices) - 1
    for j in range(N_points):
        faces.append([j, (j + 1) % N_points, end1_center_index])

    # Second end cap
    end2_center = np.array([x_fine[-1], y_fine[-1], z_fine[-1]])
    vertices = np.vstack([vertices, end2_center])
    end2_center_index = len(vertices) - 1
    for j in range(N_points):
        faces.append([(len(t_fine) - 1) * N_points + j, (len(t_fine) - 1) * N_points + (j + 1) % N_points, end2_center_index])

    # Convert faces to NumPy array
    faces = np.array(faces)

    # Convert to Trimesh object
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Export to STL
    mesh.export(f'{output_name}.stl')

    # Export to PLY
    mesh.export(f'{output_name}.ply')

    # Export to VTP (VTK PolyData)
    # Create VTK points
    # vtk_points = vtkPoints()
    # for v in vertices:
    #     vtk_points.InsertNextPoint(v.tolist())

    # # Create VTK cells
    # vtk_faces = vtkCellArray()
    # for face in faces:
    #     vtk_faces.InsertNextCell(3, face.tolist())

    # # Create VTK PolyData
    # polydata = vtkPolyData()
    # polydata.SetPoints(vtk_points)
    # polydata.SetPolys(vtk_faces)

    # # Write VTK PolyData to VTP file
    # writer = vtkPolyDataWriter()
    # writer.SetFileName('rod_mesh.vtp')
    # writer.SetInputData(polydata)
    # writer.Write()

    # print("Export completed: STL, PLY, and VTP formats.")


if __name__ == "__main__":
    
    rod1 = np.array([
        [12.556352615356445, 39.108608245849609, 182.13078308105469],
        [5.2248969078063965, 43.648529052734375, 220.02519226074219],
        [0.091214179992675781, 45.676429748535156, 253.206298828125],
        [-3.4759788513183594, 44.4451904296875, 281.99542236328125],
        [-6.6587367057800293, 43.146121978759766, 311.73406982421875]
    ])
    
    rod2 = np.array([
        [-10.730596542358398, 35.814983367919922, 178.5301513671875],
        [-16.673681259155273, 40.129436492919922, 216.92864990234375],
        [-22.284271240234375, 42.596401214599609, 250.98428344726562],
        [-27.305301666259766, 43.9449462890625, 277.98019409179688],
        [-29.183883666992188, 40.8734130859375, 309.80169677734375]
    ])
    
    generate_rod_mesh(rod1, 2, 'rod1.stl')
    generate_rod_mesh(rod2, 2, 'rod2.stl')