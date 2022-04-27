# Edge Impulse - OpenMV Object Detection Example

import sensor, image, time, os, tf, math, uos, gc, pyb

sensor.reset()                         # Reset and initialize the sensor.
sensor.set_pixformat(sensor.GRAYSCALE)    # Set pixel format to RGB565 (or GRAYSCALE)
sensor.set_framesize(sensor.QVGA)      # Set frame size to QVGA (320x240)
sensor.set_windowing((240, 240))       # Set 240x240 window.
sensor.skip_frames(time=2000)          # Let the camera adjust.

net = None
labels = None
min_confidence = 1.0
CO2_level = 0
ventilated = True

def calculate_co2_level(people_count):
	# This function should be called once every minute.
	global CO2_level
	CO2_level = CO2_level + (0.02556 * people_count)
	return CO2_level

def main():
	try:
		# Load built in model
		labels, net = tf.load_builtin_model('trained')
	except Exception as e:
		raise Exception(e)

	colors = [ # Add more colors if you are detecting more than 7 types of classes at once.
		(255,   0,   0),
		(  0, 255,   0),
		(255, 255,   0),
		(  0,   0, 255),
		(255,   0, 255),
		(  0, 255, 255),
		(255, 255, 255),
	]

	clock = time.clock()
	start = pyb.millis()
	while(True):
		clock.tick()

		img = sensor.snapshot()

		people_count = 0
		# detect() returns all objects found in the image (splitted out per class already)
		# we skip class index 0, as that is the background, and then draw circles of the center
		# of our objects
		for i, detection_list in enumerate(net.detect(img, thresholds=[(math.ceil(min_confidence * 255), 255)])):
			if (i == 0): continue # background class
			if (len(detection_list) == 0): continue # no detections for this class?

			print("********** %s **********" % labels[i])

			people_count = len(detection_list)
			print("People Count = ", people_count)

			for d in detection_list:
				[x, y, w, h] = d.rect()
				center_x = math.floor(x + (w / 2))
				center_y = math.floor(y + (h / 2))
				print('x %d\ty %d' % (center_x, center_y))
				img.draw_circle((center_x, center_y, 12), color=colors[i], thickness=2)

		# Calculate CO2 level.
		if (pyb.elapsed_millis(start) >= 60000):
			CO2_level = calculate_co2_level(people_count)
			print("Approximate CO2 Level = " + str(CO2_level) + "ounce")
			start = pyb.millis()

		print(clock.fps(), "fps", end="\n\n")

main()
