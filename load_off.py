import numpy as np


def read_off(name):
    points = []
    faces = []
    with open(name, 'r') as f:
        line = f.readline().strip()
        if line == 'OFF':
            num_verts, num_faces, num_edge = f.readline().split()
            num_verts = int(num_verts)
            num_faces = int(num_faces)
        else:
            num_verts, num_faces, num_edge = line[3:].split()
            num_verts = int(num_verts)
            num_faces = int(num_faces)

        for idx in range(num_verts):
            line = f.readline()
            point = [float(v) for v in line.split()]
            points.append(point)

        for idx in range(num_faces):
            line = f.readline()
            face = [int(t_f) for t_f in line.split()]
            faces.append(face[1:])
    triangles = np.zeros((num_faces, 3, 3)).astype(np.float32)
    for _idx, (a, b, c) in enumerate(faces):
        triangles[_idx, 0] = np.array(points[a])
        triangles[_idx, 1] = np.array(points[b])
        triangles[_idx, 2] = np.array(points[c])

    return np.expand_dims(triangles, 0)


if __name__ == '__main__':
    a = read_off('/repository/ModelNet40/airplane/test/airplane_0627.off')
    print(a.shape)
