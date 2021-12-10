'''
原理可参考https://zhuanlan.zhihu.com/p/30033898
'''
import os
import cv2
import sys
import math
import config
import collections
import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab
from scipy.linalg import lstsq
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares

num_view=10
depth_min=425.0
depth_max=2.5
rootdir = config.root
##########################
# extract corners
##########################
def extract_features(image_names):
    sift = cv2.xfeatures2d.SIFT_create(0, 3, 0.04, 10)
    key_points_for_all = []
    descriptor_for_all = []
    colors_for_all = []
    for image_name in image_names:
        image = cv2.imread(image_name)

        if image is None:
            continue
        key_points, descriptor = sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None)

        if len(key_points) <= 10:
            continue

        key_points_for_all.append(key_points)
        descriptor_for_all.append(descriptor)
        colors = np.zeros((len(key_points), 3))
        for i, key_point in enumerate(key_points):
            p = key_point.pt
            colors[i] = image[int(p[1])][int(p[0])]
        colors_for_all.append(colors)
    return np.array(key_points_for_all), np.array(descriptor_for_all), np.array(colors_for_all)


def match_features(query, train):
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn_matches = bf.knnMatch(query, train, k=2)
    matches = []
    # Apply Lowe's SIFT matching ratio test(MRT)
    for m, n in knn_matches:
        if m.distance < config.MRT * n.distance:  # ！！！！！
            matches.append(m)

    return np.array(matches)


def match_all_features(descriptor_for_all, rootdir,num_view):
    m = len(descriptor_for_all)
    matches_for_all = []
    score = np.zeros((m, m), dtype=np.float)
    score_index = np.zeros((m, num_view), dtype=np.int)
    for i in range(m):
        for j in range(i,m):
            if i==j:
                score[i][j]=0
                continue
            matches = match_features(descriptor_for_all[i], descriptor_for_all[j])
            score[i][j] = len(matches)
            matches_for_all.append(matches)
    for i in range(m):
        for j in range(i,m):
            score[j][i]=score[i][j]
    with open(rootdir+"\\text.txt", 'w') as f :
        f.write(str(m))
        f.write('\n')
        for i in range(m):
            f.write(str(i))
            f.write('\n')
            f.write(str(num_view))
            f.write(' ')
            for k in range(num_view):
                n=np.argmax(score[i])
                score_index[i][k] = n
                f.write(str(n))
                f.write(' ')
                f.write(str(score[i][n]))
                f.write(' ')
                score[i][n]=0
            f.write('\n')

    return np.array(matches_for_all),score_index


######################
# get rotations and transform between images
######################
def find_transform(K, p1, p2):
    focal_length = 0.5 * (K[0, 0] + K[1, 1])
    principle_point = (K[0, 2], K[1, 2])
    E, mask = cv2.findEssentialMat(p1, p2, focal_length, principle_point, cv2.RANSAC, 0.999, 1.0)
    cameraMatrix = np.array([[focal_length, 0, principle_point[0]], [0, focal_length, principle_point[1]], [0, 0, 1]])
    pass_count, R, T, mask = cv2.recoverPose(E, p1, p2, cameraMatrix, mask)

    return R, T, mask


def get_matched_points(p1, p2, matches):
    src_pts = np.asarray([p1[m.queryIdx].pt for m in matches])
    dst_pts = np.asarray([p2[m.trainIdx].pt for m in matches])

    return src_pts, dst_pts

def get_matched_points_inverse(p1, p2, matches):
    src_pts = np.asarray([p1[m.trainIdx].pt for m in matches])
    dst_pts = np.asarray([p2[m.queryIdx].pt for m in matches])

    return src_pts, dst_pts

def get_matched_colors(c1, c2, matches):
    color_src_pts = np.asarray([c1[m.queryIdx] for m in matches])
    color_dst_pts = np.asarray([c2[m.trainIdx] for m in matches])

    return color_src_pts, color_dst_pts

def get_matched_colors_inverse(c1, c2, matches):
    color_src_pts = np.asarray([c1[m.trainIdx] for m in matches])
    color_dst_pts = np.asarray([c2[m.queryIdx] for m in matches])

    return color_src_pts, color_dst_pts
# get overlap points
def maskout_points(p1, mask):
    p1_copy = []
    for i in range(len(mask)):
        if mask[i] > 0:
            p1_copy.append(p1[i])

    return np.array(p1_copy)


def init_structure(K, key_points_for_all, colors_for_all, matches_for_all,score_index):
    min_id=score_index[0][0]
    p1, p2 = get_matched_points(key_points_for_all[0], key_points_for_all[min_id], matches_for_all[min_id-1])
    c1, c2 = get_matched_colors(colors_for_all[0], colors_for_all[min_id], matches_for_all[min_id-1])

    if find_transform(K, p1, p2):
        R, T, mask = find_transform(K, p1, p2)
    else:
        R, T, mask = np.array([]), np.array([]), np.array([])

    #extract points
    p1 = maskout_points(p1, mask)
    p2 = maskout_points(p2, mask)
    colors = maskout_points(c1, mask)
    # reference extrinsic parameters
    R0 = np.array([[0.970263,0.00747983,0.241939],
[-0.0147429,0.999493,0.0282234],[-0.241605,-0.030951,0.969881]])
    T0 = np.array([-191.02,3.28832,22.5401])
    ex1=extri_tran(R0,T0)
    ex2=extri_tran(R,T)
    write_cam(rootdir + "\\cam_test_0.txt", K, ex1, depth_min, depth_max)
    write_cam(rootdir + "\\cam_test_1.txt", K, ex2, depth_min, depth_max)
    structure = reconstruct(K, R0, T0, R, T, p1, p2)
    rotations = [R0, R]
    motions = [T0, T]
    correspond_struct_idx = []
    for key_p in key_points_for_all:
        correspond_struct_idx.append(np.ones(len(key_p)) * - 1)
    correspond_struct_idx = np.array(correspond_struct_idx)
    idx = 0
    matches = matches_for_all[min_id-1]
    for i, match in enumerate(matches):
        if mask[i] == 0:
            continue
        correspond_struct_idx[0][int(match.queryIdx)] = idx
        correspond_struct_idx[min_id][int(match.trainIdx)] = idx
        idx += 1
    return structure, correspond_struct_idx, colors, rotations, motions


#############
# 3d reconstruct
#############
def reconstruct(K, R1, T1, R2, T2, p1, p2):
    proj1 = np.zeros((3, 4))
    proj2 = np.zeros((3, 4))
    proj1[0:3, 0:3] = np.float32(R1)
    proj1[:, 3] = np.float32(T1.T)
    proj2[0:3, 0:3] = np.float32(R2)
    proj2[:, 3] = np.float32(T2.T)
    fk = np.float32(K)
    proj1 = np.dot(fk, proj1)
    proj2 = np.dot(fk, proj2)
    s = cv2.triangulatePoints(proj1, proj2, p1.T, p2.T)
    structure = []

    for i in range(len(s[0])):
        col = s[:, i]
        col /= col[3]
        structure.append([col[0], col[1], col[2]])

    return np.array(structure)


###########################
# fusion 3d construction
###########################
def fusion_structure(matches, struct_indices, next_struct_indices, structure, next_structure, colors, next_colors):
    for i, match in enumerate(matches):
        query_idx = match.queryIdx
        train_idx = match.trainIdx
        struct_idx = struct_indices[query_idx]
        if struct_idx >= 0:
            next_struct_indices[train_idx] = struct_idx
            continue
        structure = np.append(structure, [next_structure[i]], axis=0)
        colors = np.append(colors, [next_colors[i]], axis=0)
        struct_indices[query_idx] = next_struct_indices[train_idx] = len(structure) - 1
    return struct_indices, next_struct_indices, structure, colors

def fusion_structure_inverse(matches, struct_indices, next_struct_indices, structure, next_structure, colors, next_colors):
    for i, match in enumerate(matches):
        query_idx = match.trainIdx
        train_idx = match.queryIdx
        struct_idx = struct_indices[query_idx]
        if struct_idx >= 0:
            next_struct_indices[train_idx] = struct_idx
            continue
        structure = np.append(structure, [next_structure[i]], axis=0)
        colors = np.append(colors, [next_colors[i]], axis=0)
        struct_indices[query_idx] = next_struct_indices[train_idx] = len(structure) - 1
    return struct_indices, next_struct_indices, structure, colors

def get_objpoints_and_imgpoints(matches, struct_indices, structure, key_points):
    object_points = []
    image_points = []
    for match in matches:
        query_idx = match.queryIdx
        train_idx = match.trainIdx
        struct_idx = struct_indices[query_idx]
        if struct_idx < 0:
            continue
        object_points.append(structure[int(struct_idx)])
        image_points.append(key_points[train_idx].pt)

    return np.array(object_points), np.array(image_points)

def get_objpoints_and_imgpoints_inverse(matches, struct_indices, structure, key_points):
    object_points = []
    image_points = []
    for match in matches:
        query_idx = match.trainIdx
        train_idx = match.queryIdx
        struct_idx = struct_indices[query_idx]
        if struct_idx < 0:
            continue
        object_points.append(structure[int(struct_idx)])
        image_points.append(key_points[train_idx].pt)

    return np.array(object_points), np.array(image_points)

########################
# bundle adjustment
########################

# bundle adjustment请参见https://www.cnblogs.com/zealousness/archive/2018/12/21/10156733.html

def get_3dpos(pos, ob, r, t, K):
    dtype = np.float32

    def F(x):
        p, J = cv2.projectPoints(x.reshape(1, 1, 3), r, t, K, np.array([]))
        p = p.reshape(2)
        e = ob - p
        err = e

        return err

    res = least_squares(F, pos)
    return res.x


def get_3dpos_v1(pos, ob, r, t, K):
    p, J = cv2.projectPoints(pos.reshape(1, 1, 3), r, t, K, np.array([]))
    p = p.reshape(2)
    e = ob - p
    if abs(e[0]) > config.x or abs(e[1]) > config.y:
        return None
    return pos


def bundle_adjustment(rotations, motions, K, correspond_struct_idx, key_points_for_all, structure):
    for i in range(len(rotations)):
        r, _ = cv2.Rodrigues(rotations[i])
        rotations[i] = r
    for i in range(len(correspond_struct_idx)):
        point3d_ids = correspond_struct_idx[i]
        key_points = key_points_for_all[i]
        r = rotations[i]
        t = motions[i]
        for j in range(len(point3d_ids)):
            point3d_id = int(point3d_ids[j])
            if point3d_id < 0:
                continue
            new_point = get_3dpos_v1(structure[point3d_id], key_points[j].pt, r, t, K)
            structure[point3d_id] = new_point

    return structure


#######################
# fig
#######################

# 这里有两种方式作图，其中一个是matplotlib做的，但是第二个是基于mayavi做的，效果上看，fig_v1效果更好。fig_v2是mayavi加颜色的效果。

def fig(structure, colors):
    colors /= 255
    for i in range(len(colors)):
        colors[i, :] = colors[i, :][[2, 1, 0]]
    fig = plt.figure()
    fig.suptitle('3d')
    ax = fig.gca(projection='3d')
    for i in range(len(structure)):
        ax.scatter(structure[i, 0], structure[i, 1], structure[i, 2], color=colors[i, :], s=5)
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    ax.view_init(elev=135, azim=90)
    plt.show()


def fig_v1(structure):
    mlab.points3d(structure[:, 0], structure[:, 1], structure[:, 2], mode='point', name='dinosaur')
    mlab.show()


def fig_v2(structure, colors):
    for i in range(len(structure)):
        mlab.points3d(structure[i][0], structure[i][1], structure[i][2],
                      mode='point', name='dinosaur', color=colors[i])

    mlab.show()

def write_cam(filename, intrinsic, extrinsic, depth_min, depth_max):
    with open(filename, 'w') as f:
        f.write('extrinsic\n')
        for j in range(4):
            for k in range(4):
                f.write(str(extrinsic[j, k]) + ' ')
            f.write('\n')
        f.write('\nintrinsic\n')
        for j in range(3):
            for k in range(3):
                f.write(str(intrinsic[j, k]) + ' ')
            f.write('\n')
        f.write('\n%f %f\n' % (depth_min,depth_max))
def extri_tran(R,T):
    extrinsic = np.zeros((4, 4), dtype=np.float)
    for oo in range(3):
        for pp in range(3):
            extrinsic[oo][pp] = R[oo][pp]
    #        extrinsic[:3][:3] = R
    for oo in range(3):
        extrinsic[oo][3] = T[oo]
    #        extrinsic[:3][4] = T
    extrinsic[3][:] = [0.0, 0.0, 0.0, 1.0]
    return extrinsic

def main():
    imgdir = config.image_dir
    img_names = os.listdir(imgdir)
    img_names = sorted(img_names)
    for i in range(len(img_names)):
        img_names[i] = imgdir + '\\' + img_names[i]
    # intrinsic parameters
    K = config.K

    key_points_for_all, descriptor_for_all, colors_for_all = extract_features(img_names)
    matches_for_all,score_index= match_all_features(descriptor_for_all,rootdir,num_view)
    structure, correspond_struct_idx, colors, rotations, motions = init_structure(K, key_points_for_all, colors_for_all,
                                                                                  matches_for_all,score_index)
    conlist = np.zeros(len(img_names)-1, dtype=np.int)
 #   conlist[0]=0
    num=0
    for i in range(len(img_names)-1):
        for j in range(num_view):
            n=score_index[num][j]
            if n not in conlist:
                conlist[i] = n
                num=n
                break

    num_storge=[]
    flag=0
    for kk in range(len(conlist)-1):
        num_c = 0
        if conlist[kk]<conlist[kk+1]:
            min=conlist[kk]
            max=conlist[kk+1]
        else:
            min=conlist[kk+1]
            max=conlist[kk]
            flag=1
        for jj in range(min):
            num_c+=len(img_names)-jj-1
        num_c+=max-min-1
        if  flag==1:
            num_c*=-1
        num_storge.append(num_c)
        flag=0

    for i in range(len(img_names)-2):
        if num_storge[i]>0:
            object_points, image_points = get_objpoints_and_imgpoints(matches_for_all[num_storge[i]], correspond_struct_idx[conlist[i]],
                                                                  structure, key_points_for_all[conlist[i+1]])
        else:
            object_points, image_points = get_objpoints_and_imgpoints_inverse(matches_for_all[num_storge[i]*-1],
                                                                      correspond_struct_idx[conlist[i]],
                                                                      structure, key_points_for_all[conlist[i + 1]])

        if len(image_points) < 7:
            while len(image_points) < 7:
                object_points = np.append(object_points, [object_points[0]], axis=0)
                image_points = np.appe
                、nd(image_points, [image_points[0]], axis=0)

        _, r, T, _ = cv2.solvePnPRansac(object_points, image_points, K, np.array([]))
        R, _ = cv2.Rodrigues(r)
        rotations.append(R)
        motions.append(T)
        extrinsic=extri_tran(R,T)
        write_cam(rootdir+"\\cam_test_"+str(conlist[i]+1)+".txt", K, extrinsic, depth_min, depth_max)
        if num_storge[i] > 0:
            p1, p2 = get_matched_points(key_points_for_all[conlist[i]], key_points_for_all[conlist[i+1]], matches_for_all[num_storge[i]])
            c1, c2 = get_matched_colors(colors_for_all[conlist[i]], colors_for_all[conlist[i+1]], matches_for_all[num_storge[i]])
        else:
            p1, p2 = get_matched_points_inverse(key_points_for_all[conlist[i]], key_points_for_all[conlist[i + 1]],
                                        matches_for_all[num_storge[i]*-1])
            c1, c2 = get_matched_colors_inverse(colors_for_all[conlist[i]], colors_for_all[conlist[i + 1]],
                                        matches_for_all[num_storge[i]*-1])
        next_structure = reconstruct(K, rotations[i+1], motions[i+1], R, T, p1, p2)
        if num_storge[i] > 0:
            correspond_struct_idx[conlist[i]], correspond_struct_idx[conlist[i+1]], structure, colors = fusion_structure(matches_for_all[num_storge[i]],
                                                                                                     correspond_struct_idx[
                                                                                                         conlist[i]],
                                                                                                     correspond_struct_idx[
                                                                                                         conlist[i+1]],
                                                                                                     structure,
                                                                                                     next_structure,
                                                                                                     colors, c1)
        else:
            correspond_struct_idx[conlist[i]], correspond_struct_idx[
                conlist[i + 1]], structure, colors = fusion_structure_inverse(matches_for_all[num_storge[i]*-1],
                                                                      correspond_struct_idx[
                                                                          conlist[i]],
                                                                      correspond_struct_idx[
                                                                          conlist[i + 1]],
                                                                      structure,
                                                                      next_structure,
                                                                      colors, c1)
    structure = bundle_adjustment(rotations, motions, K, correspond_struct_idx, key_points_for_all, structure)
    i = 0

    while i < len(structure):
        if math.isnan(structure[i][0]):
            structure = np.delete(structure, i, 0)
            colors = np.delete(colors, i, 0)
            i -= 1
        i += 1

    print(len(structure))
    print(len(motions))
    # np.save('structure.npy', structure)
    # np.save('colors.npy', colors)

    # fig(structure,colors)
    fig_v1(structure)
    # fig_v2(structure, colors)


if __name__ == '__main__':
    main()