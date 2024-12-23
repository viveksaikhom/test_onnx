import TI.GPIO as GPIO

GPIO.setmode(GPIO.BOARD)

pins_to_cleanup = [16, 18, 22]

for pin in pins_to_cleanup:
    GPIO.setup(pin, GPIO.OUT)

print(f'\nCleaning PINs: {", ".join(map(str, pins_to_cleanup))}')

GPIO.cleanup()

print("GPIO pins cleaned up successfully vs :).\n")
