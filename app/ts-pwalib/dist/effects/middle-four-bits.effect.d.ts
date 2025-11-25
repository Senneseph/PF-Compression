import { Effect } from '../core/effect';
/**
 * Effect that extracts the middle 4 bits from each RGB component.
 *
 * This effect:
 * 1. Extracts bits 2-5 (0-indexed) from each RGB component
 * 2. Shifts them to the right position (bits 0-3)
 * 3. Preserves the alpha channel
 */
export declare class MiddleFourBitsEffect extends Effect {
    /**
     * Initialize the Middle Four Bits Effect.
     */
    constructor();
    /**
     * Transform an ImageData object by extracting the middle 4 bits from each RGB component.
     *
     * @param imageData - The ImageData object to transform.
     * @returns The transformed ImageData object.
     */
    transform(imageData: ImageData): ImageData;
}
