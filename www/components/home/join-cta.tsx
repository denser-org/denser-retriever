import { Logo } from "@/components/ui/icons"
import Ripple from "@/components/ui/ripple"

export function JoinCTA() {
  return (
    <section className="relative rounded-sm w-[100vw]">
      <div className="relative items-center py-12 px-12 mx-auto w-full bg-white border md:px-12 lg:py-24 lg:px-24 max-w-[1400px] border-neutral-100/20 dark:bg-neutral-900 dark:border-neutral-600/30">
        <div className="relative grid grid-flow-row grid-cols-2 md:grid-cols-3">
          <div className="flex items-center justify-center col-span-2 md:col-span-1">
            <div className="w-1/3 p-8 rounded-full bg-neutral-950 dark:border md:w-fit">
              <Logo className="text-white w-60 h-60" />
            </div>
          </div>
          <div className="flex flex-col items-stretch justify-center w-full col-span-2 gap-4">
            <h5 className="mt-8 text-4xl font-semibold leading-none text-left lg:text-5xl dark:text-white text-neutral-600 font-display">
              DenserRetriever v1 Beta coming!
            </h5>
            <p className="mt-3 text-base leading-relaxed text-left text-gray-500 dark:text-neutral-100">
              You can try out DenserRetriever in your self-hosted machine,
              with an extremely simple docker setup.
            </p>
            <div className="flex w-1/2 gap-4 mt-6">
              <a
                href="https://retriever.denser.ai/docs/install/install-server"
                target="_blank"
                className="flex items-center justify-center w-full px-10 py-4 text-2xl text-center text-white transition duration-500 ease-in-out transform rounded-md focus:ring-2 focus:ring-offset-2 focus:ring-offset-white focus:outline-none bg-neutral-600 font-display group dark:focus:ring-offset-neutral-900 dark:focus:ring-neutral-700 hover:bg-neutral-700 focus:ring-neutral-100"
              >
                Deploy now
                <div className="hidden lg:block w-0 translate-x-[100%] pl-0 opacity-0 transition-all duration-200 lg:group-hover:w-5 lg:group-hover:translate-x-0 lg:group-hover:pl-4 lg:group-hover:opacity-100">
                  <svg
                    className="size-6"
                    xmlns="http://www.w3.org/2000/svg"
                    viewBox="0 0 256 256"
                  >
                    <rect width="256" height="256" fill="none" />
                    <path
                      d="M80,128a80,80,0,1,1,80,80H72A56,56,0,1,1,85.92,97.74"
                      fill="none"
                      stroke="currentColor"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="16"
                    />
                  </svg>
                </div>
              </a>
            </div>
          </div>
        </div>

        <Ripple />
      </div>
    </section>
  )
}
