class TrackableObject:
	def __init__(self, objectID, centroid, area_id):
		# store the object ID, then initialize a list of centroids
		# using the current centroid
		self.objectID = objectID
		self.centroids = [centroid]

		# initialize a boolean used to indicate if the object has
		# already been counted or not
		self.counted = False

		# Which side the object was seen / Area bit
		self.area_id = area_id
