import { Background } from "@/components/background";
import { buttonVariants } from "@/components/ui/button";

const Home = () => {
  return (
    <div className="w-full">
      <Background />
      <div className="container relative h-screen px-6 m-auto md:px-12 lg:px-7">
        <div className="py-40 mx-auto lg:py-56 md:w-9/12 lg:w-7/12 dark:lg:w-6/12">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white md:text-6xl lg:text-4xl xl:text-6xl">
            Next Generation <br />
            <span className="text-blue-500">AI Retriever.</span>
          </h1>
          <p className="mt-8 text-gray-700 dark:text-gray-300">
            Lorem ipsum dolor sit amet consectetur adipisicing elit. Odio
            incidunt nam itaque sed eius modi error totam sit illum. Voluptas
            doloribus asperiores quaerat aperiam. Quidem harum omnis beatae
            ipsum soluta!
          </p>
          <div className="flex flex-col mt-16 space-y-2 md:flex-row lg:space-y-0 md:w-max sm:space-x-6">
            <button
              type="button"
              title="Get started"
              className={buttonVariants({
                size: "lg",
              })}
            >
              Get started
            </button>
            <button
              type="button"
              title="About us"
              className={buttonVariants({
                size: "lg",
                variant: "ghost",
              })}
            >
              Contact us
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;
