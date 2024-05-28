import { Logo } from "./ui/icons"

export function Footer() {
  return (
    <footer className="py-12 mt-auto border-t text-secondary-foreground dark:bg-neutral-800">
      <div className="px-8 mx-auto max-w-7xl md:px-12 lg:px-32">
        <div className="xl:grid xl:grid-cols-3 xl:gap-8">
          <p className="text-2xl font-bold uppercase">
            <Logo className="size-10" />
          </p>

          <div className="grid grid-cols-2 gap-8 mt-12 text-sm font-medium text-gray-500 lg:grid-cols-3 lg:mt-0 xl:col-span-2">
            <div>
              <h3 className="text-lg text-neutral-800 dark:text-neutral-100">
                Information
              </h3>
              <ul role="list" className="mt-4 space-y-2">
                <li>
                  <a
                    href="https://github.com/denser-org/denser-retriever/blob/main/LICENSE"
                    target="_blank"
                    className="relative after:absolute after:-bottom-1 after:left-0 after:h-[2px] after:w-full after:origin-bottom-right after:scale-x-0 after:bg-gray-500 after:transition-transform after:duration-200 after:ease-[cubic-bezier(0.65_0.05_0.36_1)] hover:after:origin-bottom-left hover:after:scale-x-100 p-0.5 rounded-md transition duration-300 outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-white dark:focus:ring-offset-neutral-800 dark:focus:ring-neutral-600 focus:ring-neutral-100"
                  >
                    License
                  </a>
                </li>
              </ul>
            </div>
            <div>
              <h3 className="text-lg text-neutral-800 dark:text-neutral-100">
                Links
              </h3>
              <ul role="list" className="mt-4 space-y-2">
                <li>
                  <a
                    href="https://github.com/denser-org/denser-retriever"
                    target="_blank"
                    className="relative after:absolute after:-bottom-1 after:left-0 after:h-[2px] after:w-full after:origin-bottom-right after:scale-x-0 after:bg-gray-500 after:transition-transform after:duration-200 after:ease-[cubic-bezier(0.65_0.05_0.36_1)] hover:after:origin-bottom-left hover:after:scale-x-100 p-0.5 rounded-md transition duration-300 outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-white dark:focus:ring-offset-neutral-800 dark:focus:ring-neutral-600 focus:ring-neutral-100"
                  >
                    GitHub
                  </a>
                </li>
                <li>
                  <a
                    href="https://github.com/denser-org/denser-retriever/issues"
                    target="_blank"
                    className="relative after:absolute after:-bottom-1 after:left-0 after:h-[2px] after:w-full after:origin-bottom-right after:scale-x-0 after:bg-gray-500 after:transition-transform after:duration-200 after:ease-[cubic-bezier(0.65_0.05_0.36_1)] hover:after:origin-bottom-left hover:after:scale-x-100 p-0.5 rounded-md transition duration-300 outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-white dark:focus:ring-offset-neutral-800 dark:focus:ring-neutral-600 focus:ring-neutral-100"
                  >
                    Issues
                  </a>
                </li>
                <li>
                  <a
                    href="mailto:support@denser.ai?Subject=DenserRetriever%20Feedback"
                    target="_blank"
                    className="relative after:absolute after:-bottom-1 after:left-0 after:h-[2px] after:w-full after:origin-bottom-right after:scale-x-0 after:bg-gray-500 after:transition-transform after:duration-200 after:ease-[cubic-bezier(0.65_0.05_0.36_1)] hover:after:origin-bottom-left hover:after:scale-x-100 p-0.5 rounded-md transition duration-300 outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-white dark:focus:ring-offset-neutral-800 dark:focus:ring-neutral-600 focus:ring-neutral-100"
                  >
                    Email
                  </a>
                </li>
              </ul>
            </div>
            <div className="mt-12 md:mt-0">
              <h3 className="text-lg text-neutral-800 dark:text-neutral-100">
                About
              </h3>
              <ul role="list" className="mt-4 space-y-2">
                <li>
                  <a
                    href="https://denser.ai"
                    target="_blank"
                    className="relative after:absolute after:-bottom-1 after:left-0 after:h-[2px] after:w-full after:origin-bottom-right after:scale-x-0 after:bg-gray-500 after:transition-transform after:duration-200 after:ease-[cubic-bezier(0.65_0.05_0.36_1)] hover:after:origin-bottom-left hover:after:scale-x-100 p-0.5 rounded-md transition duration-300 outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-white dark:focus:ring-offset-neutral-800 dark:focus:ring-neutral-600 focus:ring-neutral-100"
                  >
                    Denser AI
                  </a>
                </li>
              </ul>
            </div>
          </div>
        </div>
        <div className="flex flex-col pt-12 md:flex-row md:justify-between md:items-center">
          <span className="flex items-center justify-between w-full text-sm font-medium text-gray-500">
            <div className="text-xl font-light font-display">
              <span className="">2024</span>
              <a
                aria-label="denser"
                href="https://denser.ai"
                className="p-0.5 mx-2 text-pink-400 rounded-md transition duration-300 outline-none hover:text-pink-600 focus:ring-2 focus:ring-offset-2 focus:ring-offset-white dark:focus:ring-offset-neutral-800 dark:focus:ring-neutral-600 focus:ring-neutral-100"
              >
                denser.ai
              </a>
            </div>
            <kbd className="py-0.5 px-1.5 text-[0.65rem] font-medium rounded-sm shadow-[rgba(250,_250,_250,_0.2)_0px_0px_0px_1px,_rgba(250,_250,_250,_0.2)_0px_2px_0px_0px] dark:shadow-neutral-400 shadow-neutral-200">
              v1.0.0 Beta
            </kbd>
          </span>
        </div>
      </div>
    </footer>
  )
}
