import numpy as np
import cv2
import cv2.aruco

def get_Dictionary(markers_list):
    # https://datigrezzi.com/2019/07/12/custom-aruco-markers-in-python/
    markers = np.array(markers_list, dtype=np.uint8)
    number = markers.shape[0]
    side = markers.shape[1]
    assert side == markers.shape[2]

    aruco_dict = cv2.aruco.custom_dictionary(0, side, 1)

    size_test = cv2.aruco.Dictionary_getByteListFromBits(np.zeros((side,side), dtype=np.uint8))
    aruco_dict.bytesList = np.empty(shape = (number, size_test.shape[1], size_test.shape[2]), dtype = np.uint8)

    for i,data in enumerate(markers):
        aruco_dict.bytesList[i] = cv2.aruco.Dictionary_getByteListFromBits(data)

    aruco_dict.maxCorrectionBits = markers_min_distance(markers_list)

    return aruco_dict

def markers_min_distance(markers):
    distances = []
    for m1 in markers:
        for m2 in markers:
            m1 = np.array(m1)
            m2 = np.array(m2)
            if((m1 == m2).all()):
                continue
            distances.append( np.sum(m1^m2) )
    # cast to int is required. np.int64 can mess stuff up
    return int(min(distances))

def print_markers(markers):
    for marker in markers:
        for row in marker:
            print(row)
        print()

def add_black_border(markers):
    out = []
    size = len(markers[0][0])
    for marker in markers:
        assert len(marker[0]) == size
        new_marker = []
        new_marker.append( [0]*(size +2) )
        new_marker.extend(  [ [0] + row + [0] for row in marker] )
        new_marker.append( [0]*(size +2) )
        out.append(new_marker)
    return out

markers = []
#post legs:
markers.append( [[1,1,0,1,1], [1,1,0,1,1], [1,0,1,0,1], [1,1,1,1,1], [1,1,1,1,1]] )
markers.append( [[1,1,0,1,1], [1,1,0,1,1], [1,0,1,0,1], [0,0,1,1,0], [1,1,1,0,1]] )
markers.append( [[1,1,0,1,1], [1,1,0,1,1], [1,0,1,0,1], [1,0,1,1,0], [1,0,1,1,0]] )

#leg4
markers.append( [[1,1,0,1,1], [1,1,0,1,1], [1,0,1,0,1], [0,1,1,1,1], [1,0,1,0,0]] )
markers.append( [[1,1,0,1,1], [1,1,0,1,1], [1,0,1,0,1], [0,1,1,1,0], [0,1,1,1,0]] )

#leg5
markers.append( [[1,1,0,1,1], [1,1,0,1,1], [1,0,1,0,1], [1,0,1,1,1], [0,1,1,0,0]] )
markers.append( [[1,1,0,1,1], [1,1,0,1,1], [1,0,1,0,1], [0,0,1,1,1], [0,0,1,1,1]] )

#leg6
markers.append( [[1,1,0,1,1], [1,1,0,1,1], [1,0,1,0,1], [1,1,1,1,0], [0,0,1,0,1]] )
markers.append( [[1,1,0,1,1], [1,1,0,1,1], [1,0,1,0,1], [0,0,1,0,1], [1,1,1,1,0]] )

#leg7
markers.append( [[1,1,0,1,1], [1,1,0,1,1], [1,0,1,0,1], [1,1,1,0,0], [1,1,1,0,0]] )
markers.append( [[1,1,0,1,1], [1,1,0,1,1], [1,0,1,0,1], [0,1,1,0,0], [1,0,1,1,1]] )

#add two wide border (vs one wide border? why?)
markers = add_black_border(markers)
# markers = add_black_border(markers)


dictionary = get_Dictionary(markers)

# tags used by urc not in standard dictionary
# dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)


mtx = None
dist = None

# isight
# dist =  np.array([[ 9.10040511e-01],
#        [ 1.09754967e+01],
#        [-4.87192834e-03],
#        [-1.85610734e-02],
#        [ 3.96624729e+01],
#        [ 1.49320854e+00],
#        [ 8.84809151e+00],
#        [ 4.57254128e+01],
#        [ 0.00000000e+00],
#        [ 0.00000000e+00],
#        [ 0.00000000e+00],
#        [ 0.00000000e+00],
#        [ 0.00000000e+00],
#        [ 0.00000000e+00]])

# mtx =  np.array([[1.08479191e+03, 0.00000000e+00, 3.56409835e+02],
#        [0.00000000e+00, 1.08479191e+03, 3.45422188e+02],
#        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

#c920

dist =  np.array([[ 2.68013365e+00],
       [ 4.45921553e+01],
       [-1.66943228e-02],
       [-1.41329716e-02],
       [-2.14441609e+02],
       [ 2.64990348e+00],
       [ 4.34424237e+01],
       [-2.10288217e+02],
       [ 0.00000000e+00],
       [ 0.00000000e+00],
       [ 0.00000000e+00],
       [ 0.00000000e+00],
       [ 0.00000000e+00],
       [ 0.00000000e+00]])
mtx =  np.array([[671.14819063,   0.        , 348.3507712 ],
       [  0.        , 671.14819063, 204.40567262],
       [  0.        ,   0.        ,   1.        ]])

def main():
    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Display the resulting frame
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(frame, dictionary)
        image = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        if mtx is not None:
            size_of_marker =  0.025 # side lenght of the marker in meter
            rvecs,tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(corners, size_of_marker , mtx, dist)

            if tvecs is not None:
                length_of_axis = 0.1
                for i in range(len(tvecs)):
                    image = cv2.aruco.drawAxis(image, mtx, dist, rvecs[i], tvecs[i], length_of_axis)
                print('rotation', rvecs[0])
                print('translation', tvecs[0])

        cv2.imshow('frame', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
