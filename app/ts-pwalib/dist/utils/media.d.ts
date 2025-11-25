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
export declare function requestMediaAccess(options: MediaOptions): Promise<MediaStream>;
/**
 * Get a list of available media devices.
 *
 * @returns A promise that resolves to an array of MediaDeviceInfo objects.
 */
export declare function getMediaDevices(): Promise<MediaDeviceInfo[]>;
/**
 * Get a list of available video devices.
 *
 * @returns A promise that resolves to an array of MediaDeviceInfo objects representing video devices.
 */
export declare function getVideoDevices(): Promise<MediaDeviceInfo[]>;
/**
 * Get a list of available audio devices.
 *
 * @returns A promise that resolves to an array of MediaDeviceInfo objects representing audio devices.
 */
export declare function getAudioDevices(): Promise<MediaDeviceInfo[]>;
/**
 * Create a video element from a MediaStream.
 *
 * @param stream - The MediaStream to use.
 * @returns A video element with the stream as its source.
 */
export declare function createVideoElement(stream: MediaStream): HTMLVideoElement;
