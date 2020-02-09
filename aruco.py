import numpy as np
import cv2
import cv2.aruco

def get_Dictionary(markers):
    # https://datigrezzi.com/2019/07/12/custom-aruco-markers-in-python/
    markers = np.array(markers, dtype=np.uint8)
    number = markers.shape[0]
    side = markers.shape[1]
    assert side == markers.shape[2]

    aruco_dict = cv2.aruco.custom_dictionary(0, side, 1)

    size_test = cv2.aruco.Dictionary_getByteListFromBits(np.zeros((side,side), dtype=np.uint8))
    aruco_dict.bytesList = np.empty(shape = (number, size_test.shape[1], size_test.shape[2]), dtype = np.uint8)

    for i,data in enumerate(markers):
        aruco_dict.bytesList[i] = cv2.aruco.Dictionary_getByteListFromBits(data)

    return aruco_dict

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
markers = add_black_border(markers)


# for marker in markers:
#     for row in marker:
#         print(row)
#     print()

dictionary = get_Dictionary(markers)

# tags used by urc not in standard dictionary
# dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)

def main():
    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(frame, dictionary)
        image = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        cv2.imshow('frame', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
