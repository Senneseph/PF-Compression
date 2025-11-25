import { Effect } from '../core/effect';
/**
 * Effect that inverts the colors of an image.
 *
 * This effect:
 * 1. Inverts each RGB component (255 - value)
 * 2. Preserves the alpha channel
 */
export declare class ColorNegativeEffect extends Effect {
    /**
     * Initialize the Color Negative Effect.
     */
    constructor();
    /**
     * Transform an ImageData object by inverting its colors.
     *
     * @param imageData - The ImageData object to transform.
     * @returns The transformed ImageData object with inverted colors.
     */
    transform(imageData: ImageData): ImageData;
}
