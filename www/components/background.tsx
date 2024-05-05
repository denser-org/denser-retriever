import Image from "next/image";

export const Background = () => (
  <div className="absolute top-0 left-0 right-0 -z-10">
    <Image
      src="/images/background.svg"
      alt="Background"
      width={1800}
      height={100}
      aria-hidden="true"
    />
  </div>
);
