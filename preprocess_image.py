import cv2
import numpy as np
 
 
MAX_FEATURES = 10000
GOOD_MATCH_PERCENT = 0.2
 
 
def alignImages(im1, im2):

  # Convert images to grayscale
  #print(im2.shape)
  im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
  im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
   
  # Detect ORB features and compute descriptors.
  orb = cv2.ORB_create(MAX_FEATURES)
  keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
  keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
   
  # Match features.
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches = matcher.match(descriptors1, descriptors2, None)
   
  # Sort matches by score
  matches.sort(key=lambda x: x.distance, reverse=False)
 
  # Remove not so good matches
  numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
  matches = matches[:numGoodMatches]
 
  # Draw top matches
  imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
  #cv2.imwrite("hoadon2/match-align-a.jpg", imMatches)
   
  # Extract location of good matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)
 
  for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt
   
  # Find homography
  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

  # Use homography
  height, width, channels = im2.shape

  im1Reg = cv2.warpPerspective(im1, h, (width, height))

  return im1Reg, h
 
def alignFile(filename1, filename2):
  #print(filename1, filename2)
  imReference = cv2.imread(filename1, cv2.COLOR_BGR2RGB)
  im = cv2.imread(filename2, cv2.COLOR_BGR2RGB)
  imReg, h = alignImages(im, imReference)
  #cv2.imshow("aaa",imReg)
  #cv2.waitKey()
  outFilename = filename2#.replace('.png','_align.png')
  cv2.imwrite(outFilename, imReg)
  return outFilename