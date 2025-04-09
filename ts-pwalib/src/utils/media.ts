/**
 * Utility functions for working with media devices.
 */

/**
 * Options for accessing media devices.
 */
export interface MediaOptions {
  video?: boolean | MediaTrackConstraints;
  audio?: boolean | MediaTrackConstraints;
}

/**
 * Request access to media devices.
 * 
 * @param options - Options for accessing media devices.
 * @returns A promise that resolves to a MediaStream object.
 * @throws Error if media devices cannot be accessed.
 */
export async function requestMediaAccess(options: MediaOptions): Promise<MediaStream> {
  try {
    return await navigator.mediaDevices.getUserMedia(options);
  } catch (error) {
    console.error('Error accessing media devices:', error);
    throw new Error('Could not access media devices. Please check permissions.');
  }
}

/**
 * Get a list of available media devices.
 * 
 * @returns A promise that resolves to an array of MediaDeviceInfo objects.
 */
export async function getMediaDevices(): Promise<MediaDeviceInfo[]> {
  try {
    // Ensure permissions are granted before enumerating devices
    await navigator.mediaDevices.getUserMedia({ audio: true, video: true });
    return await navigator.mediaDevices.enumerateDevices();
  } catch (error) {
    console.error('Error enumerating media devices:', error);
    return [];
  }
}

/**
 * Get a list of available video devices.
 * 
 * @returns A promise that resolves to an array of MediaDeviceInfo objects representing video devices.
 */
export async function getVideoDevices(): Promise<MediaDeviceInfo[]> {
  const devices = await getMediaDevices();
  return devices.filter(device => device.kind === 'videoinput');
}

/**
 * Get a list of available audio devices.
 * 
 * @returns A promise that resolves to an array of MediaDeviceInfo objects representing audio devices.
 */
export async function getAudioDevices(): Promise<MediaDeviceInfo[]> {
  const devices = await getMediaDevices();
  return devices.filter(device => device.kind === 'audioinput');
}

/**
 * Create a video element from a MediaStream.
 * 
 * @param stream - The MediaStream to use.
 * @returns A video element with the stream as its source.
 */
export function createVideoElement(stream: MediaStream): HTMLVideoElement {
  const video = document.createElement('video');
  video.srcObject = stream;
  video.autoplay = true;
  return video;
}
