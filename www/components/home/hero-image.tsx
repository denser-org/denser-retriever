"use client"

import Image from "next/image"
import Screenshot from "../../public/images/screenshot-hero.png"
import ScreenshotLight from "../../public/images/screenshot-hero.png"

export const HeroImage = () => {
  return (
    <div className="hidden overflow-hidden absolute right-0 top-12 p-px max-w-screen-sm rounded-md shadow-md transition duration-300 ease-in-out lg:block hover:shadow-xl w-[30vw] rotate-[9deg] group gradient-23 dark:gradient-33 hover:translate-y-[2vw] hover:rotate-[3deg] hover:scale-[1.3]">
      <div className="absolute inset-0 rotate-45 glow w-[100px] h-[100px]"></div>
      <Image
        className="relative inline-block transition bg-black bg-cover rounded-md opacity-0 dark:p-1 dark:opacity-100 gradient-border object-fit size-0 dark:size-fit"
        src={Screenshot}
        sizes="(max-width: 640px) 300px, (max-width: 960px) 500px, 30vw"
        quality={100}
        priority
        alt="Screenshot of SvelteKasten"
      />
      <Image
        className="relative inline-block p-1 transition bg-gray-100 bg-cover rounded-md opacity-100 dark:p-0 dark:opacity-0 gradient-border object-fit size-fit dark:size-0"
        src={ScreenshotLight}
        sizes="(max-width: 640px) 300px, (max-width: 960px) 500px, 40vw"
        quality={100}
        priority
        alt="Screenshot of SvelteKasten"
      />
      <div className="absolute inset-0 flex h-full w-full justify-center [transform:skew(-12deg)_translateX(-100%)] group-hover:duration-1000 group-hover:[transform:skew(-12deg)_translateX(100%)]">
        <div className="relative w-12 h-full bg-gray-100/90 blur-md dark:bg-white/20"></div>
      </div>
    </div>
  )
}
