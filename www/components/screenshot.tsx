import Image from "next/image";
import { cn } from "@/utils/cn";

type ScreenshotProps = {
  src: string;
  alt: string;
  full: boolean;
  className: string;
};

export function Screenshot({
  src,
  alt,
  full = false,
  className,
}: ScreenshotProps) {
  return (
    <div className="w-full">
      <div
        className={cn(
          "mx-auto mt-6 flex max-w-fit max-h-min justify-center overflow-hidden rounded-lg border dark:border-neutral-800 border-zinc-200 shadow-sm dark:shadow-lg dark:bg-neutral-950 bg-zinc-100 p-4",
          full ? "w-full max-w-full" : "",
          className,
        )}
      >
        <Image
          src={src}
          alt={alt}
          className={cn(
            "m-0 object-cover object-top block w-full h-full select-none rounded-lg outline-red-500",
            full ? "w-full" : "ring-2 dark:ring-neutral-800 ring-gray-200",
          )}
        />
      </div>
    </div>
  );
}
