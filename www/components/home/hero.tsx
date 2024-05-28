import { HeroText } from "@/components/home/hero-text"
import { HeroImage } from "@/components/home/hero-image"

export function Hero() {
  return (
    <div className="flex relative flex-col gap-4 items-center px-2 w-full md:flex-row lg:px-8">
      <HeroText />
      <HeroImage />
    </div>
  )
}
