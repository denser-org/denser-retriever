import { Plus } from "@phosphor-icons/react/dist/ssr";

export function HeroText() {
  return (
    <div className="flex flex-col w-full md:w-auto">
      <div className="relative flex flex-col w-full max-w-screen-sm mt-8 overflow-hidden">
        <div className="flex justify-start hero">
          <h1 className="relative p-2 tracking-tighter text-[13vw] sm:text-[12vw] md:text-[clamp(5rem,_4vw_+_1rem,_6rem)]">
            <span className="px-1 overflow-visible">Cutting-Edge</span>
            <span className="px-1 overflow-visible">AI Retriever</span>
            <span className="px-1 overflow-visible">for RAG</span>
            <Plus className="absolute right-0 size-12 bottom-[2.5rem]" />
            <span className="mt-6 w-[110%] border-b-4 border-neutral-600 dark:border-neutral-500"></span>
          </h1>
        </div>
      </div>
    </div>
  );
}
